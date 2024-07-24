# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it 
# under the terms of the GNU Affero General Public License 
# as published by the Free Software Foundation, either version 3 
# of the License, or any later version. This program is distributed 
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License 
# for more details. You should have received a copy of the Lesser GNU 
# General Public License along with this program.  
# If not, see <http://www.gnu.org/licenses/>.

# Written on top of local package wandb==0.12.1 and remote api 0.13.3
# "wandb login" in a terminal before using this script
# Runs states : finished, running, crashed
# Important run params/functions : .id, .state, .files, .config, .summary, .history(), .file('filename.ext').download(), api.run(f"{ENTITY}/{PROJECT}/{run.id}")

import hydra
import os
import pandas as pd
import wandb

from omegaconf import DictConfig
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

from common import logger

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'models')


class Downloader:
    def __init__(self, entity, project):
        self.entity = entity
        self.project = project
        self.api = wandb.Api(timeout=25)
        self.runs_url = f'https://wandb.ai/{entity}/{project}/runs/'

    def get_runs(self, query_filter = None):
        finished, running, other = [], [], []

        logger.info("Loading Runs DB...")
        if query_filter:
            finished_query = self.api.runs(f"{self.entity}/{self.project}", {
                "display_name": {"$regex": f".*{query_filter}.*"}
            }, per_page = 4)
        else:
            finished_query = self.api.runs(f"{self.entity}/{self.project}", per_page = 4)
        finished_query[0]
        l = finished_query.length

        logger.info("Loading Runs to memory...")
        for i in tqdm(range(l)):
            run = finished_query[i]
            if run.state == 'finished':
                finished += [run]
            elif run.state == 'running':
                running += [run]
            else:
                other += [run]
        return finished, running, other

    def check_runs(self, runs, query_filter):
        logger.info('Checking list of finished runs...')
        finished, _, _ = self.get_runs(query_filter)
        for run in finished:
            if run.id in runs:
                runs[runs.index(run.id)] = run
            if run.name in runs:
                runs[runs.index(run.name)] = run

        runs_found = []
        for run in runs:
            if type(run).__name__ == 'str':
                logger.info(f'Could not find {run}')
            else:
                runs_found += [run]
        return runs_found

    def download_model(self, run, download_dir = DEFAULT_DOWNLOAD_DIR):
        download_location = os.path.join(download_dir, f'{run.id}.{run.name}.pt')
        if os.path.exists(download_location):
            logger.info(f"Skipping : '{run.id}.{run.name}.pt' already exists")
        else:
            try:
                model_artifact = self.api.artifact(f'{self.entity}/{self.project}/run_{run.id}_model:best')
                model_artifact.download(download_dir)
                if os.path.exists(os.path.join(download_dir, 'best.pt')):
                    os.rename(os.path.join(download_dir, 'best.pt'), download_location)
                else:
                    logger.info('Skipping : Could not download.')
            except Exception as e:
                logger.info(f'Could not download {run}. E:{e}')

    def save_summary(self, runs, download_dir = DEFAULT_DOWNLOAD_DIR, project = 'results'):
        summary_list, config_list, name_list = [], [], []
        for run in runs:
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k,v in run.config.items()
                if not k.startswith('_')})

            name_list.append(run.name)

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })

        save_path = os.path.join(download_dir, f'{project}.csv')
        logger.info(f'Saving WANDB summary to: {save_path}')
        runs_df.to_csv(save_path)


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    download_params = cfg['ml']['wandb']['download']
    entity = cfg['ml']['wandb']['entity']
    project = cfg['ml']['wandb']['project']
    downloader = Downloader(entity, project)

    query_filter = None if not download_params['query_filter'] else download_params['query_filter']
    finished, running, other = downloader.get_runs(query_filter)

    runs_to_process = []
    if 'runs' in download_params:
        runs = downloader.check_runs(download_params['runs'], query_filter)
        runs_to_process = runs
    elif download_params['list_all']:
        for run in finished + running + other:
            runs_to_process += [run]
    elif download_params['list_finished']:
        runs_to_process = finished
    elif download_params['list_running']:
        runs_to_process = running

    downloaded_nb = 0
    runs_to_print = []
    runs_to_download = []

    folder = os.path.join(*download_params['folder'], project)
    folder = os.path.abspath(folder)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    folder = str(folder)

    for run in runs_to_process:
        download_location = os.path.join(folder, f'{run.id}.{run.name}.pt')
        downloaded = True
        if os.path.exists(download_location):
            downloaded = 'Yes'
            downloaded_nb += 1
        else:
            downloaded = 'No'
            runs_to_download += [run]

        runs_to_print += [[run.id, run.name, run.state, downloaded, downloader.runs_url + run.id]]

    logger.info(tabulate(runs_to_print, headers=['ID', 'NAME', 'STATUS', 'DOWNLOADED', 'URL']))

    if download_params['list_all']:
        logger.info(f'Models: {len(finished + running + other)}, Downloaded: {downloaded_nb} (In : {folder}). Can not download {len(running)}, still running.')
    elif download_params['list_finished'] or 'runs' in download_params:
        logger.info(f'Models: {len(finished)}, Downloaded: {downloaded_nb} (In : {folder})')
    elif download_params['list_running']:
        logger.info(f'Models: {len(running)}. Still running. Not able to download weights yet.')

    if download_params['download'] and runs_to_download:
        for run in runs_to_download:
            logger.info(f'Downloading : {run.id}.{run.name}.pt')
            downloader.download_model(run, folder)

    try:
        logger.info('Saving Runs summary !')
        downloader.save_summary(runs_to_process, folder, project)
    except Exception as e:
        logger.info('Could not save summary.')
        logger.info(e)


if __name__ == "__main__":
    main()

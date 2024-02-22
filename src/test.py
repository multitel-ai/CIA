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
import json
import sys
import csv
import ast

from omegaconf import DictConfig
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

from common import logger, create_yaml_file

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOWNLOAD_DIR = os.path.join(BASE_PATH, 'models')


# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO


def get_weights(weights_path, project):
    weights = weights_path.glob('*.pt')
    return weights

def build_runs_info(weights_folder, project, weights):
    summary_file = weights_folder / f'{project}.csv'
    summary_processed = []
    weights = list(weights)
    
    with open(summary_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  

        for row in reader:
            # Split the CSV row into individual fields
            id, dict1_str, dict2_str, name = row

            # Convert the dictionary strings to dictionaries
            dict1 = ast.literal_eval(dict1_str)
            dict2 = ast.literal_eval(dict2_str)

            for w in weights:
                run_id = w.name.split('.')[0]
                run_name = '.'.join(w.name.split('.')[1:-1])
                if run_name == name:
                    info_dict = {
                        # 'id': int(id),
                        'run_id': run_id,
                        'run_name': name,
                        'epochs': dict2['epochs'],
                        'data_size': dict2['data_size'],
                        'control_net': dict2['control_net'],
                        'sampling': dict2['sampling'],
                        'map': 0,
                        # 'data': dict2['data'],
                        'weights': str(w.absolute()),
                    }

                    summary_processed += [info_dict]

    return summary_processed


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data = cfg['data']
    base_path = Path(data['base'])
    test_path = base_path / 'test'

    download_params = cfg['ml']['wandb']['download']
    entity = cfg['ml']['wandb']['entity']
    project = cfg['ml']['wandb']['project']

    folder = os.path.join(*download_params['folder'], project)
    folder = os.path.abspath(folder)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    results_file = folder / "results.csv"


    weights = get_weights(folder, project)
    new_runs_info = build_runs_info(folder, project, weights)

    if os.path.isfile(results_file):
        # read older runs
        with open(results_file, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            next(reader)
            # Create an empty list to store the rows as dictionaries
            runs_info = []
            
            # Loop through each row in the CSV file
            for row in reader:
                # Append each row as a dictionary to the list
                runs_info.append(row)

        ids = [r['run_id'] for r in runs_info]

        for run in new_runs_info:
            if run['run_id'] not in ids:
                runs_info += [run]
            
    else:
        runs_info = new_runs_info


    # create yaml test file
    test_yaml_file = test_path / 'data.yaml'
    create_yaml_file(test_yaml_file, test_path, test_path, test_path)


    # test newer runs
    for i, run in enumerate(runs_info):
        run['map'] = float(run['map'])
        if run['map'] != 0:
            print(f'[{i}/{len(runs_info)}]: Skipping {run["run_name"]}, already tested with an map of {run["map"]}')
            continue

        print(f'[{i}/{len(runs_info)}]: Testing {run["run_name"]}, already tested with an map of {run["map"]}')
        model = YOLO(run['weights'])
        metrics = model.val(data = str(test_yaml_file.absolute()))
        run['map'] = metrics.box.map
        
        with open(results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(run.keys())
            for run in runs_info:
                writer.writerow(run.values())


if __name__ == "__main__":
    main()

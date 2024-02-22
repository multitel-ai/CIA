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

import hydra
import os
import sys
import uuid

from pathlib import Path
from omegaconf import DictConfig 


# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data = cfg['data']
    base_path = Path(data['base']) 
    
    if cfg['ml']['augmentation_percent']==0:  
        REAL_DATA = Path(base_path) / data['real'] 
    else:
        fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
        REAL_DATA = Path(base_path) / data['real'] / fold
    
    data_yaml_path = REAL_DATA / 'data.yaml'

    model = YOLO("yolov8n.yaml")
    cn_use = cfg['model']['cn_use']
    aug_percent = cfg['ml']['augmentation_percent']
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"
    sampling_code_name = (cfg['ml']['sampling']['metric'] + '_' + cfg['ml']['sampling']['sample']) 

    model.train(
        data = str(data_yaml_path.absolute()),
        epochs = cfg['ml']['epochs'],
        entity = cfg['ml']['wandb']['entity'],
        project = cfg['ml']['wandb']['project'],
        name = name,
        control_net = 'Starting_point' if cfg['ml']['augmentation_percent'] == 0 else cn_use,
        sampling = sampling_code_name if cfg['ml']['sampling']['enable'] else 'disabled'
    )


if __name__ == '__main__':
    main()

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
    GEN_DATA_PATH =  Path(base_path) / data['generated'] / cfg['model']['cn_use']
    
    if cfg['ml']['augmentation_percent']==0:  
        REAL_DATA = Path(base_path) / data['real'] 
    else:
        fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
        REAL_DATA = Path(base_path) / data['real'] / fold
    
    run = "" if cfg['ml']['augmentation_percent']==0 else cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
    data_yaml_path = REAL_DATA / 'data.yaml'

    model = YOLO("yolov8n.yaml")
    cn_use = cfg['model']['cn_use']
    aug_percent = cfg['ml']['augmentation_percent']
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"
    sampling_code_name = (cfg['ml']['sampling']['metric'] + '_' + cfg['ml']['sampling']['type']) 

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

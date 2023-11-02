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
    aug_percent = cfg['model']['augmentation_percent']
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"

    model.train(
        data = str(data_yaml_path.absolute()),
        epochs = cfg['ml']['epochs'],
        project = cfg['ml']['wandb']['project'],
        name = name
    )


if __name__ == '__main__':
    main()

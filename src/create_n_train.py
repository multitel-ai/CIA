import hydra
import os
import sys
import uuid
import glob
import random
import yaml

from omegaconf import DictConfig
from pathlib import Path
from typing import List

from create_dataset import create_mixte_dataset, list_images, create_yaml_file


# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data = cfg['data']
    base_path = Path(data['base'])
    GEN_DATA_PATH =  Path(base_path) / data['generated'] / cfg['model']['cn_use']
    REAL_DATA_PATH = Path(base_path) / data['real'] 
    
    if cfg['ml']['augmentation_percent']==0:  
        fold = REAL_DATA_PATH
    else:
        fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent']) 
        fold = REAL_DATA_PATH / fold
        
    if not os.path.isdir(fold): 
        os.makedirs(fold)

    create_mixte_dataset(
        REAL_DATA_PATH, 
        GEN_DATA_PATH, 
        fold, 
        cfg['ml']['augmentation_percent'],
        cfg['ml']['train_nb'],
        cfg['ml']['val_nb'],
        cfg['ml']['test_nb'],
    )

    
    data_yaml_path = fold / 'data.yaml'

    model = YOLO("yolov8n.yaml")
    cn_use = cfg['model']['cn_use']
    aug_percent = cfg['model']['augmentation_percent']
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"

    model.train(
        data = str(data_yaml_path.absolute()),
        epochs = cfg['ml']['epochs'],
        project = 'sdcn',
        name = name
    )


if __name__ == '__main__':
    main()
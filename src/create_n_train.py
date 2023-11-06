import hydra
import os
import sys
import uuid
import glob
import random
import yaml

from omegaconf import DictConfig, open_dict
from pathlib import Path

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
    
    # Add Sampling Path
    IQA_PATH = Path(base_path) / 'iqa'
    file_json_iqa = IQA_PATH / f"{cfg['model']['cn_use']}_iqa.json"
    with open_dict(cfg):
        cfg['ml']['sampling']['score_file'] = file_json_iqa

    if cfg['ml']['augmentation_percent'] == 0:  
        fold = REAL_DATA_PATH
    else:
        fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent']) 
        fold = REAL_DATA_PATH / fold

    data_yaml_path = create_mixte_dataset(
        base_path,
        REAL_DATA_PATH, 
        GEN_DATA_PATH, 
        fold, 
        cfg['ml']['augmentation_percent'],
        cfg['ml']['train_nb'],
        cfg['ml']['val_nb'],
        cfg['ml']['test_nb'],
        cfg['ml']['sampling'],
    )

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
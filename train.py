import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import uuid


sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics import YOLO

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = cfg['data_path']
    base_path = os.path.join(*data_path['base'])
    REAL_DATA_PATH = Path(base_path) / data_path['real']
    data_yaml_path = REAL_DATA_PATH / 'data.yaml'

    

    model = YOLO("yolov8n.yaml")
    model.train(
        data = str(data_yaml_path.absolute()), 
        epochs = cfg['ml']['epochs'],
        project = 'sdcn',
        # entity = 'sdcn-nantes',
        name = f"{uuid.uuid4().hex.upper()[0:6]}_{cfg['model']['cn_use']}_{cfg['model']['augmentation_percent']}"
    )

if __name__ == '__main__':
    main()
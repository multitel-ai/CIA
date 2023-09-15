import hydra
import os
import sys
import uuid

from pathlib import Path
from omegaconf import DictConfig

# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(sys.path[0], "ultralytics"))
from ultralytics import YOLO


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data = cfg['data']
    base_path = Path(data['base'])
    REAL_data = Path(base_path) / data['real']
    data_yaml_path = REAL_data / 'data.yaml'

    model = YOLO("yolov8n.yaml")
    cn_use = cfg['model']['cn_use']
    aug_percent = cfg['model']['augmentation_percent']
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"

    model.train(
        data = str(data_yaml_path.absolute()),
        epochs = cfg['ml']['epochs'],
        project = 'sdcn',
        # entity = 'sdcn-nantes',
        name = name
    )


if __name__ == '__main__':
    main()

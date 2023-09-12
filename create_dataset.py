import os
import argparse
import random
import yaml
from pathlib import Path
from logger import logger
import hydra
from omegaconf import DictConfig
from typing import List
import glob

def create_mixte_dataset(real_images_dir: str, synth_images_dir: str, txt_dir: str, per_synth_data: float, n_file: int = 0, formats: List[str] = ['jpg', 'png', 'jpeg']):
    """
    Construct the txt file containing a percentage of real and synthetic data
    :param real_images_dir: path to the folder containing real images
    :param synth_images_dir: path to the folder containing synthetic images
    :param txt_dir: path used to create the txt file
    :param per_synth_data: float, [0, 1], percentage of synthetic data compared to real ones
    :param n_file: int, number of the file used for the txt and yaml names
    :return: /
    """
    train_txt_path = txt_dir / 'train.txt'
    val_txt_path = txt_dir / 'val.txt'
    test_txt_path = txt_dir / 'test.txt'
    data_yaml_path = txt_dir / 'data.yaml'

    real_images_path = real_images_dir / 'images'
    val_images_path = Path(str(real_images_path.absolute()).replace('/real/', '/val/'))
    test_images_path = Path(str(real_images_path.absolute()).replace('/real/', '/test/'))
    
    real_images = list_images(real_images_path, formats)
    synth_images = list_images(synth_images_dir, formats)
    val_images = list_images(val_images_path, formats)
    test_images = list_images(test_images_path, formats)
    
    # shuffle images
    random.Random(42).shuffle(real_images)
    random.Random(42).shuffle(synth_images)

    # nb_real_images = int(len(real_images) * (1 - per_synth_data))
    nb_synth_images = int(len(real_images) * per_synth_data)

    real_images = real_images
    synth_images = synth_images[:nb_synth_images]

    train_images = real_images + synth_images

    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(train_images))

    with open(val_txt_path, 'w') as f:
        f.write('\n'.join(val_images))

    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_images))

    
    create_yaml_file(
        data_yaml_path, 
        train_txt_path,
        val_txt_path,
        test_txt_path
    )


def list_images(images_path: Path, formats: List[str]):
    images = []
    for format in formats:
        images += [
            *glob.glob(str(images_path.absolute()) + f'/*.{format}')
        ]
    return images


def create_yaml_file(save_path: Path, train: Path, val: Path, test: Path):
    """
    Construct the yaml file
    :param txt_dir: path used to create the txt files
    :param yaml_dir: path used to create the yaml file
    :param n_file: int, number of the file used for the txt and yaml names
    :return: /
    """
    yaml_file = {
        'train': str(train.absolute()),
        'val': str(val.absolute()),
        'test': str(test.absolute()),
        'names': {0: 'person'}
    }
    
    with open(save_path, 'w') as file:
        yaml.dump(yaml_file, file)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    data_path = cfg['data_path']
    base_path = os.path.join(*data_path['base'])
    REAL_DATA_PATH = Path(base_path) / data_path['real']

    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']

    create_mixte_dataset(REAL_DATA_PATH, GEN_DATA_PATH, REAL_DATA_PATH, cfg['ml']['augmentation_percent'])


if __name__ == '__main__':
    main()
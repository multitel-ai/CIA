import glob
import hydra
import os
import random
import yaml
import json

from omegaconf import DictConfig, open_dict
from pathlib import Path
from typing import List, Tuple

def create_mixte_dataset(base_path: str,
                         real_images_dir: str,
                         synth_images_dir: str,
                         txt_dir: str,
                         per_synth_data: float,
                         train_nb:int,
                         val_nb:int,
                         test_nb:int,
                         sample: dict,
                         formats: List[str] = ['jpg', 'png', 'jpeg'],
                         seed:int = 42):
    """
    Construct the txt file containing a percentage of real and synthetic data

    :param str real_images_dir: path to the folder containing real images
    :param str synth_images_dir: path to the folder containing synthetic images
    :param str txt_dir: path used to create the txt file
    :param float per_synth_data: percentage of synthetic data compared to real (ranges in [0, 1])

    :return: None
    :rtype: NoneType
    """

    if sample['enable']:
        txt_dir = txt_dir / (sample['metric'] + '_' + sample['sample'])

    if not os.path.isdir(txt_dir): 
        os.makedirs(txt_dir)
    
    train_txt_path = txt_dir / 'train.txt'
    val_txt_path = txt_dir / 'val.txt'
    test_txt_path = txt_dir / 'test.txt'
    data_yaml_path = txt_dir / 'data.yaml'

    real_images_path = real_images_dir / 'images'
    val_images_path = Path(base_path) / 'val' / 'images'
    test_images_path = Path(base_path) / 'test' / 'images'
    
    real_images = list_images(real_images_path, formats, train_nb)
    val_images = list_images(val_images_path, formats, val_nb)
    test_images = list_images(test_images_path, formats, test_nb)

    if sample['enable']:
        with open(sample['score_file'], 'r') as f:
            score_data = json.load(f)

        # set sort direction of synthetic images to work with best or worst
        if sample['sample'] == 'best':
            order = score_data['best'][sample['metric']]
        else:
            if score_data['best'][sample['metric']] == 'smaller':
                order = 'bigger'
            else:
                order = 'smaller'

        synth_images, scores = sort_based_on_score(
            score_data['image_paths'], 
            score_data[sample['metric']],
            order
        )
        synth_images = [str((Path(base_path).parent / img).absolute()) for img, score in zip(synth_images, scores)]
        # for i in range(synth_images.__len__()):
        #     print(synth_images[i], scores[i])
    else:
        synth_images = list_images(synth_images_dir, formats)
        # shuffle images
        random.Random(seed).shuffle(synth_images)
    
    # shuffle images
    random.Random(seed).shuffle(real_images)

    # nb_real_images = int(len(real_images) * (1 - per_synth_data))
    nb_synth_images = int(len(real_images) * per_synth_data)
    synth_images = synth_images[:nb_synth_images]

    train_images = real_images + synth_images

    create_files_list(train_images, train_txt_path)
    create_files_list(val_images, val_txt_path)
    create_files_list(test_images, test_txt_path)

    create_yaml_file(data_yaml_path, train_txt_path, val_txt_path, test_txt_path)

    return data_yaml_path

def sort_based_on_score(image_paths: List[str], scores: List[float], direction: str = 'smaller') -> Tuple[List[str], List[int]]:
    """
    Sorts two arrays of the same size based on scores and returns sorted scores and image paths.

    Args:
        image_paths (List[str]): List of image paths.
        scores (List[int]): List of scores for each image.
        direction (str): either ascending or descending

    Returns:
        Tuple[List[str], List[int]]: A tuple containing the sorted scores and sorted image paths.
    """
    # Combine scores and image paths into a list of tuples
    combined_data = list(zip(scores, image_paths))
    # Sort the combined data based on scores (ascending order)
    sorted_data = sorted(combined_data, key = lambda x: x[0], reverse = False if direction == 'smaller' else True)
    # Extract sorted scores and image paths
    sorted_scores, sorted_image_paths = zip(*sorted_data)
    return sorted_image_paths, sorted_scores


def create_files_list(image_files, txt_file_path):
    with open(txt_file_path, 'w') as f:
        f.write('\n'.join(image_files))


def list_images(images_path: Path, formats: List[str], limit:int = None):
    images = []
    for format in formats:
        images += [
            *glob.glob(str(images_path.absolute()) + f'/*.{format}')
        ]
    return images[:limit]


def create_yaml_file(save_path: Path, train: Path, val: Path, test: Path):
    """
    Construct the yaml file

    :param pathlib.Path txt_dir: path used to create the txt files
    :param pathlib.Path yaml_dir: path used to create the yaml file

    :return: None
    :rtype: NoneType
    """

    yaml_file = {
        'train': str(train.absolute()),
        'val': str(val.absolute()),
        'test': str(test.absolute()),
        'names': {0: 'person'}
    }

    with open(save_path, 'w') as file:
        yaml.dump(yaml_file, file)


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg : DictConfig) -> None:
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


if __name__ == '__main__':
    main()

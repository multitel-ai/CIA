import gdown
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
from pathlib import Path
import zipfile

from omegaconf import DictConfig
import hydra
from PIL import Image

# We are hard coding this because of the origin of the data. No need to
# put it in the configuration file as for this use case it will not change.
data_dir = 'home/ahmadh/Coco_1FullPerson'
url = 'https://drive.google.com/uc?id=1scqiUhjrB1SrYowWljbtz1Tf_gvvEgsY'
data_type='train2017'


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    REAL_DATA_PATH = Path(cfg['data_path']['real'])

    # Define where coco will live
    # This path will change after getting the data
    COCO_DATA_PATH = REAL_DATA_PATH / 'coco'

    COCO_DATA_PATH.mkdir(parents=True, exist_ok=True)
    data_zip_path = COCO_DATA_PATH / 'Coco_1FullPerson.zip'

    if not os.path.exists(data_zip_path):
        gdown.download(url, str(data_zip_path), quiet=False)

    if not os.path.exists(COCO_DATA_PATH / data_dir):
        with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(COCO_DATA_PATH))

    COCO_DATA_PATH = COCO_DATA_PATH / data_dir
    image_paths = [COCO_DATA_PATH / image_path for image_path in os.listdir(COCO_DATA_PATH)]

    im = Image.open(image_paths[44])
    print(image_paths[44])
    im.show()

if __name__ == '__main__':
    main()
import cv2
import hydra
import os
import wget
import zipfile

from omegaconf import DictConfig
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm

from logger import logger


def cocobox2yolo(img_path, coco_box):
    I = cv2.imread(img_path)
    image_hight, image_width = I.shape[0:2]

    [left, top, box_width, box_hight] = coco_box
    x_center = (left + box_width / 2) / image_width
    y_center = (top + box_hight / 2) / image_hight

    box_width /= image_width
    box_hight /= image_hight
    yolo_box = [x_center, y_center, box_width, box_hight]

    return yolo_box


def download_coco(data_path: Path,
                  coco_bbx: str = 'Coco_1FullPerson_bbx',
                  coco_caps: str = 'Coco_1FullPerson_caps',
                  image_path: str = 'Coco_1FullPerson',
                  annotations_path: str = 'annotations',
                  data_zip_name: str = 'Coco_1FullPerson.zip',
                  annotations_zip_name: str = 'annotations_trainval2017.zip'):

    image_path: Path = data_path / image_path
    annotations_path: Path = data_path / annotations_path
    bbx_path: Path = data_path / coco_bbx
    caps_path: Path = data_path / coco_caps

    dirs = data_path, image_path, annotations_path, bbx_path, caps_path
    logger.info(f'Attempting to create directories {[str(d) for d in dirs]}')
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    path_to_data_zip = data_path / data_zip_name
    path_to_annotations_zip = data_path / annotations_zip_name

    data_url = 'https://transfer.sh/FW8MjTFjDv/Coco_1FullPerson.zip'
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    # Only perform the work if necessary
    if not os.path.exists(path_to_data_zip):
        logger.info(f'Downloading zip images from {data_url} to {path_to_data_zip}')
        wget.download(data_url, out=str(path_to_data_zip))
    if not len(os.listdir(image_path)):
        logger.info(f'Extracting zip images to {image_path}')
        with zipfile.ZipFile(path_to_data_zip, 'r') as zip_ref:
            zip_ref.extractall(str(data_path))

    if not os.path.exists(path_to_annotations_zip):
        logger.info(f'Downloading zip annotations from {annotations_url} to {path_to_annotations_zip}')
        wget.download(annotations_url, out=str(path_to_annotations_zip))
    if not len(os.listdir(annotations_path)):
        logger.info(f'Extracting zip annotations to {annotations_path}')
        with zipfile.ZipFile(path_to_annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(str(data_path))

    return image_path, annotations_path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Get all paths
    data_path = cfg['data_path']
    REAL_DATA_PATH = Path(data_path['base']) / data_path['real']
    COCO_PATH = REAL_DATA_PATH / 'coco'

    REAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    COCO_PATH.mkdir(parents=True, exist_ok=True)

    # Download if necessary
    image_path, annotations_path = download_coco(COCO_PATH)
    coco_version = 'train2017'

    annFile = annotations_path / f'instances_{coco_version}.json'
    annFile_keypoints = annotations_path / f'person_keypoints_{coco_version}.json'
    annFile_captions = annotations_path / f'captions_{coco_version}.json'

    coco = COCO(annFile.absolute())
    coco_keypoints = COCO(annFile_keypoints.absolute())
    coco_captions = COCO(annFile_captions.absolute())

    catIds = coco.getCatIds(catNms=['person'])
    all_images = list(image_path.glob('*.jpg'))

    logger.info(f'Writting captions and boxes info ...')
    for img_path in tqdm(all_images):
        img_path = str(img_path.absolute())
        img_id = int(img_path.split('/')[-1].split('.jpg')[0])
        Keypoints_annIds = coco_keypoints.getAnnIds(imgIds = img_id, catIds = catIds, iscrowd = None)
        Keypoints_anns = coco_keypoints.loadAnns(Keypoints_annIds)

        caps_annIds = coco_captions.getAnnIds(imgIds = img_id)
        caps_anns = coco_captions.loadAnns(caps_annIds)

        bbox_text_path = img_path.replace('.jpg', '.txt').replace('Coco_1FullPerson','Coco_1FullPerson_bbx')
        captions_text_path = img_path.replace('.jpg', '.txt').replace('Coco_1FullPerson','Coco_1FullPerson_caps')

        with open(bbox_text_path, 'w') as file:
            coco_box = Keypoints_anns[0]['bbox']
            yolo_box = cocobox2yolo(img_path,coco_box)
            KP_Yolo_format = '0 '+' '.join(list(map(str, yolo_box)))
            file.write(KP_Yolo_format)

        with open(captions_text_path, 'w') as file:
            captions = [caps['caption'] for caps in caps_anns]
            file.write('\n'.join(captions))


if __name__ == "__main__":
   main()

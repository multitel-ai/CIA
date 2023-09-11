import cv2
import hydra

from omegaconf import DictConfig
from pathlib import Path
from pycocotools.coco import COCO


def cocobox2yolo(img_path, coco_box):
	I = cv2.imread(img_path)
	Image_hight, Image_width = I.shape[0:2]

	[left, top, box_width, box_hight] = coco_box
	x_center = (left + box_width / 2) / Image_width
	y_center = (top + box_hight / 2) / Image_hight

	box_width /= Image_width
	box_hight /= Image_hight
	Yolobbx = [x_center, y_center, box_width, box_hight]

	return Yolobbx


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
	dataset_folder = 'coco'
	coco_version = 'train2017'

	data_path = cfg['data_path']
	REAL_DATA_PATH = Path(data_path['base']) / data_path['real']
	DATASETS_DATA_PATH = Path(data_path['datasets'])

	# Specify the results path
	(REAL_DATA_PATH).mkdir(parents=True, exist_ok=True)

	(DATASETS_DATA_PATH).mkdir(parents=True, exist_ok=True)
	coco_path = DATASETS_DATA_PATH / dataset_folder / 'Coco_1FullPerson'
	coco_annotations_path = DATASETS_DATA_PATH / dataset_folder / 'annotations'

	annFile = coco_annotations_path / f'instances_{coco_version}.json'
	annFile_keypoints = coco_annotations_path / f'person_keypoints_{coco_version}.json'
	annFile_captions = coco_annotations_path / f'captions_{coco_version}.json'

	coco = COCO(annFile.absolute())
	coco_keypoints = COCO(annFile_keypoints.absolute())
	coco_captions = COCO(annFile_captions.absolute())

	catIds = coco.getCatIds(catNms=['person'])

	all_images = list(coco_path.glob('*.jpg'))
	print(all_images[0])

	for img_path in all_images:
		img_path = str(img_path.absolute())
		img_id = int(img_path.split('/')[-1].split('.jpg')[0])
		Keypoints_annIds = coco_keypoints.getAnnIds(imgIds = img_id, catIds = catIds, iscrowd = None)
		Keypoints_anns = coco_keypoints.loadAnns(Keypoints_annIds)

		caps_annIds = coco_captions.getAnnIds(imgIds = img_id);
		caps_anns = coco_captions.loadAnns(caps_annIds)

		bbox_text_path = img_path.replace('.jpg', '.txt').replace('Coco_1FullPerson','Coco_1FullPerson_bbx')

		captions_text_path = img_path.replace('.jpg', '.txt').replace('Coco_1FullPerson','Coco_1FullPerson_caps')

		with open(bbox_text_path, 'w') as file:
			coco_box = Keypoints_anns[0]['bbox']
			Yolo_bbx = cocobox2yolo(img_path,coco_box)
			KP_Yolo_format = '0 '+' '.join(list(map(str, Yolo_bbx)))
			file.write(KP_Yolo_format)

		with open(captions_text_path, 'w') as file:
			captions = [caps['caption'] for caps in caps_anns]
			file.write('\n'.join(captions))


if __name__ == "__main__":
   main()

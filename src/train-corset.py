import hydra
import os
import sys
import uuid
import yaml
import wandb
import numpy as np
from pathlib import Path
from omegaconf import DictConfig 
import torch
from tqdm import tqdm

from coreset_utils import query

# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
	data = cfg['data']
	base_path = Path(data['base']) 
	GEN_DATA_PATH =  Path(base_path) / data['generated'] / cfg['model']['cn_use']

	if cfg['ml']['augmentation_percent'] == 0 or cfg['active']["abled"]:
		REAL_DATA = Path(base_path) / data['real']
	else:
		fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
		REAL_DATA = Path(base_path) / data['real'] / fold
	
	data_yaml_path = REAL_DATA / 'data.yaml'
	
	cn_use = cfg['model']['cn_use']
	aug_percent = cfg['model']['augmentation_percent']
	name_ = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"
	sampling_code_name = 'coreset'

	if cfg['active']["abled"]:
		print("ACTIVE LEARNING : Rounds") # ask about this

		model_path = Path(base_path) / str( cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent']) + ".pt")

		for i in range(cfg['active']['rounds']):
			name = "_".join(name_.split("_")[:-1]) + f"_active_{i}"
			model = YOLO("yolov8n.yaml")

			if i == 0: 
				torch.save({"model": model.model.cpu()}, model_path)
				train = data_yaml_path.parent / f"active_{cn_use}"
				
				if os.path.isdir(train):
					os.system(f"rm {str(train.absolute())}/*")
			  

			elif i > 0 and os.path.exists(model_path): # great new directory for the new coreset loop
				model.model = torch.load(model_path)['model']
				model.model.query = False
				model.predictor = None

				data_yaml_path_2 = Path(str(data_yaml_path.absolute()).replace(".yaml", "_" + cn_use + ".yaml"))
				fold = data_yaml_path_2.parent + f"active_{cn_use}"
				print("fold", fold)

				data_yaml_path_2 = fold / data_yaml_path_2.name
				print(f"data_yaml_path_2 is on iteration {i}", data_yaml_path_2) 

				if not os.path.isdir(fold):
					os.makedirs(fold)
					
				if not os.path.exists(data_yaml_path_2):
					os.system(f"cp {(data_yaml_path.absolute())} {(data_yaml_path_2.absolute())}") 

				data_yaml_path = data_yaml_path_2

				print("Querying CORESET...")
				real_images_list = []
				for im in sorted(os.listdir(Path(base_path) / data['real'] / "coco/Coco_1FullPerson") ):
					if im not in os.listdir(Path(base_path) / data['real'] / "images"):
						real_images_list += [Path(base_path) / data['real'] / "coco/Coco_1FullPerson" / im ]

				generated_images_list = []
				for im in sorted(os.listdir(GEN_DATA_PATH)):
					generated_images_list += [GEN_DATA_PATH / im]

				query(
						model,  # _coreset
						real_images_list,  # on rajoute a chaque fois des images réelles?
						generated_images_list, 
						str(data_yaml_path),
						cfg['active']['sel'],
						base_path.parent,
						f"active_{cn_use}", 
						i
					)
				print("Queried CORESET Successfully")

			print("data_yaml_path", data_yaml_path)
			model = YOLO("yolov8n.yaml")
			model.train(
				data = data_yaml_path,
				epochs = cfg['ml']['epochs'],
				entity = cfg['ml']['wandb']['entity'],
				project = cfg['ml']['wandb']['project'],
				name = name,
				control_net = f"active_{cn_use}",
				sampling = sampling_code_name,
			)

			if wandb.run is not None:
				wandb.run.finish()

			torch.save({"model":model.model.cpu()}, model_path)


	elif cfg['ml']['baseline']:
		print("ACTIVE LEARNING : Baseline") # ask about this 
		cn_use = "coreset-baseline"
		aug_percent = cfg['ml']['augmentation_percent_baseline']
		name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"

		data_yaml_path_2 = Path(str(data_yaml_path.absolute()).replace(".yaml", "_" + cn_use +'_' + str(aug_percent) + ".yaml"))
		
		fold = data_yaml_path_2.parent + f"{cn_use}_{aug_percent}"
		data_yaml_path_2 = fold / data_yaml_path_2.name
	   
		if not os.path.isdir(fold):
			os.makedirs(fold)

		if not os.path.exists(data_yaml_path_2):
			os.system(f'cp {(data_yaml_path.absolute())} {(data_yaml_path_2.absolute())}')
	   
		with open(data_yaml_path, 'r') as f:
			data = yaml.safe_load(f) 

		old_train = train = data["train"]

		if fold not in train:
			print("Fixing fold path in data['train']")
			train = data["train"].replace("train", (fold.parent / 'train').absolute)
			print("train", train)
			
		data_yaml_file = data
		data_yaml_file["train"] = train

		with open(data_yaml_path_2, 'w') as f:
			yaml.dump(data, f)

		with open(old_train, 'r') as f: # train plutot que old_train
			used_data = f.readlines()
			used_data = [data.replace('\n','') for data in used_data]
			print("used data", used_data)
	

		# Processing used_data
		sel = int(cfg['ml']['train_nb'] * aug_percent) # 250 * ask about this
		if sel < len(used_data):
			used_data_processed = used_data + used_data[:sel] # ask about this
		else:
			used_data_processed = used_data + int(aug_percent) * used_data
		print("Size of used_data_processed", len(used_data_processed))

		with open(train, 'w') as f: 
			f.writelines(used_data_processed)

		data_yaml_path = data_yaml_path_2
			
			
	print("Start Training with data :", data_yaml_path)
	model = YOLO("yolov8n.yaml")
	model.model.query = False
	model.train(
		data = data_yaml_path,
		epochs = cfg['ml']['epochs'],
		entity = cfg['ml']['wandb']['entity'],
		project = cfg['ml']['wandb']['project'],
		name = name_,
		control_net = f"active_{cn_use}",
		sampling = sampling_code_name,
	)


if __name__ == '__main__':
	main()

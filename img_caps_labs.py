import json
import os
import shutil
import gdown
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import glob
import hydra


dataDir='/content/home/ahmadh/'
dataType='train2017'

x = glob.glob('{}/Coco_1FullPerson/*.jpg'.format(dataDir))

Coco_1FullPerson =[]
for file in x:
  Coco_1FullPerson += [int(file.split('/')[-1].split('.')[0])]

#load annotations
annFile='{}annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

annFile_kps = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps = COCO(annFile_kps)

annFile_caps = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile_caps)

catIds = coco.getCatIds(catNms=['person'])

import matplotlib.pyplot as plt

def get_annotation(img_id, caption = True, show= False):
  img_path = '{}Coco_1FullPerson/{}.jpg'.format(dataDir,str(img_id).zfill(12))

  # load caption annotations
  Keypoints_annIds = coco_kps.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
  Keypoints_anns = coco_kps.loadAnns(Keypoints_annIds)

  if caption:
    # load caption annotations
    caps_annIds = coco_caps.getAnnIds(imgIds=img_id);
    caps_anns = coco_caps.loadAnns(caps_annIds)

  if show:
    # load and display keypoints annotations
    I = io.imread(img_path)
    # display keypoints annotations
    plt.imshow(I); plt.axis('off')
    ax = plt.gca()
    coco_kps.showAnns(Keypoints_anns)
    if caption:
      # display caption annotations
      coco_caps.showAnns(caps_anns)

  if caption: return (Keypoints_anns, caps_anns)
  else: return (Keypoints_anns)

def save_data(img_id):
    
    img_path = '{}Coco_1FullPerson/{}.jpg'.format(dataDir, str(img_id).zfill(12))
    new_img_path = os.path.join(dataDir, "images", "{}.jpg".format(str(img_id).zfill(12)))
    shutil.copy(img_path, new_img_path)
    
    Keypoints_annIds = coco_kps.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
    Keypoints_anns = coco_kps.loadAnns(Keypoints_annIds)
    keypoints_str = ""
    for ann in Keypoints_anns:
        keypoints_str += ' '.join(map(str, ann['keypoints'])) + "\n"
    with open(os.path.join(dataDir, "labels", "{}.txt".format(str(img_id).zfill(12))), 'w') as file:
        file.write(keypoints_str)
    
    caps_annIds = coco_caps.getAnnIds(imgIds=img_id)
    caps_anns = coco_caps.loadAnns(caps_annIds)
    all_captions = [cap['caption'] for cap in caps_anns]
    with open(os.path.join(dataDir, "captions", "{}.json".format(str(img_id).zfill(12))), 'w') as file:
        json.dump(all_captions, file)
    


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main():

    folders = ["images", "captions", "labels"]
    for folder in folders:
        path = os.path.join(dataDir, folder)
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
   main()
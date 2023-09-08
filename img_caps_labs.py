import json
import os
import shutil

img_path = os.path.join(dataDir, 'images')
os.makedirs(img_path, exist_ok=True)

labs_path = os.path.join(dataDir, 'f_labels')
os.makedirs(labs_path, exist_ok=True)

caps_path = os.path.join(dataDir, 'f_captions')
os.makedirs(caps_path, exist_ok=True)

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
mkdir -p datasets/coco

# download images
wget https://transfer.sh/FW8MjTFjDv/Coco_1FullPerson.zip -O datasets/coco/Coco_1FullPerson.zip
unzip datasets/coco/Coco_1FullPerson.zip -d datasets/coco

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O datasets/coco/annotations_trainval2017.zip
unzip datasets/coco/annotations_trainval2017.zip -d datasets/coco

mkdir -p datasets/coco/Coco_1FullPerson_bbx
mkdir -p datasets/coco/Coco_1FullPerson_caps

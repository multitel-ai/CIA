import os
import sys

sys.path.append(os.path.join(sys.path[0], "ultralytics"))

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")

# Use the model
model.train(data = "./bank/coco.yaml", epochs = 300)  # train the model
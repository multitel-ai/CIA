import cv2
import numpy as np
from PIL import Image

from controlnet_aux import OpenposeDetector


# !!!
# All extractors have the same api: extract(img: Image) -> Image

class Canny:
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        
        canny_image = Image.fromarray(image)
        return canny_image


class OpenPose:
    def __init__(self, model: str = "lllyasviel/ControlNet"):
        self.model = OpenposeDetector.from_pretrained(model)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        pose = self.model(image)
        return pose

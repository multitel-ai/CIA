from PIL import Image
import numpy as np
import cv2
from controlnet_aux import OpenposeDetector

class Canny:
    def __init__(self):
        pass

    def canny(image: Image, low_threshold: int = 100, high_threshold: int = 200) -> Image:
        low_threshold = 100
        high_threshold = 200
            
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        
        return canny_image


class OpenPose:
    def __init__(self, model: str = "lllyasviel/ControlNet"):
        self.model = OpenposeDetector.from_pretrained(model)

    def detect(self, image: np.array):
        pose = self.model(image)
        return pose
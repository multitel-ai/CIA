import cv2
import numpy as np
from PIL import Image

from controlnet_aux import OpenposeDetector


# !!!
# All extractors have the same api: extract(img: Image) -> Image

class Canny:
    def __init__(self, auto_threshold: bool = False, low_threshold: int = 100, high_threshold: int = 200) -> None:
        self.auto_threshold = auto_threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def canny_get_thresholds(self, image: np.array) -> float:

        '''
        Args: image; numpy array of RGB image
        Returns: low; float; the lower threshold value for canny edge detector
                 high; float; the upper threshold value for canny edge detector
        '''
  
        img_median = np.median(image)
        img_std = np.std(image)
        low = int(max(0, img_median - 0.5*img_std))
        high = int(min(255, img_median + 0.5*img_std))
        return low, high

    def extract(self, image: Image) -> Image:
        
        '''
        Arg: image; Image; image in pillow format
        Returns: canny_image; Image; image with edges marked
        '''
        image = np.array(image)
        
        if self.auto_threshold:
            low_threshold, high_threshold = self.canny_get_thresholds(image)
        else: 
            low_threshold, high_threshold = self.low_threshold, self.high_threshold
        
        image = cv2.Canny(image, low_threshold, high_threshold)
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

# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it 
# under the terms of the GNU Affero General Public License 
# as published by the Free Software Foundation, either version 3 
# of the License, or any later version. This program is distributed 
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License 
# for more details. You should have received a copy of the Lesser GNU 
# General Public License along with this program.  
# If not, see <http://www.gnu.org/licenses/>.

import cv2
import numpy as np
import torch

from common import draw_landmarks_on_image
from controlnet_aux import OpenposeDetector
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MpImage
from mediapipe import ImageFormat
from PIL import Image
from torchvision.transforms import ToPILImage
from typing import Tuple
from controlnet_aux.processor import Processor
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "ultralytics"))
from ultralytics import YOLO


AVAILABLE_EXTRACTORS = ('openpose', 'canny', 'mediapipe_face', 'segmentation')


def extract_model_from_name(raw_name: str) -> str:
    if 'openpose' in raw_name:
        return 'openpose'
    elif 'canny' in raw_name:
        return 'canny'
    elif 'mediapipe' in raw_name:
        return 'mediapipe_face'
    elif 'false_segmentation' in raw_name:
        return 'false_segmentation'
    elif 'segmentation' in raw_name:
        return 'segmentation'
    else:
        raise Exception(f'Unkown model: {raw_name}')


class Extractor:
    def __new__(cls, control_model: str, **kwargs):
        if control_model not in AVAILABLE_EXTRACTORS:
            raise Exception(f'Unknown control model: {control_model}')

        if 'openpose' in control_model:
            return OpenPose(**kwargs)
        elif 'canny' in control_model:
            return Canny(**kwargs)
        elif 'mediapipe_face' in control_model:
            return MediaPipeFace(**kwargs)
        elif 'false_segmentation' in control_model: # for paper
            return FalseSegmentation(**kwargs)
        elif 'segmentation' in control_model:
            return Segmentation(**kwargs)


class FalseSegmentation:
    def __init__(self, **kwargs):
        self.model = YOLO("yolov8m-seg.pt")

    def extract(self, image: Image) -> Image:
        image = np.array(image)

        results = self.model.predict(image)
        result = results[0].masks[0].data[0]
        
        seg_image = torch.t(result)
        seg_image = result[None, None, ...] # ToPILImage()(result[None, :])
        seg_image = torch.concat((seg_image, seg_image, seg_image), axis=1) 
        
        return seg_image
        
class Segmentation:
    def __init__(self, **kwargs):
        self.model = YOLO("yolov8m-seg.pt")

    def extract(self, image: Image) -> Image:
        image = np.array(image)

        seg_image = self.model.predict(image)
        seg_image = seg_image[0].masks[0].data[0]
        
        # seg_image = torch.t(result)
        seg_image = torch.stack(3 * (seg_image,)) # 
        seg_image = ToPILImage()(seg_image)
        
        return seg_image


class Canny:
    def __init__(self,
                 auto_threshold: bool = False,
                 low_threshold: int = 100,
                 high_threshold: int = 200,
                 **kwargs):
        self.auto_threshold = auto_threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def canny_get_thresholds(self, image: np.array) -> Tuple[float, float]:

        """
        Args: image; numpy array of RGB image
        Returns: low; float; the lower threshold value for canny edge detector
                 high; float; the upper threshold value for canny edge detector
        """

        img_median = np.median(image)
        img_std = np.std(image)
        low = int(max(0, img_median - 0.5*img_std))
        high = int(min(255, img_median + 0.5*img_std))
        return low, high

    def extract(self, image: Image) -> Image:

        """
        Arg: image; Image; image in pillow format
        Returns: canny_image; Image; image with edges marked
        """

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

    def __str__(self) -> str:
        return 'Extractor(canny)'


class OpenPose:
    def __init__(self, model: str = 'lllyasviel/ControlNet', **kwargs):
        self.model = OpenposeDetector.from_pretrained(model)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        pose = self.model(image)
        return pose

    def __str__(self) -> str:
        return 'Extractor(openpose)'


class MediaPipeFace:
    def __init__(self, **kwargs):
        pass

    def extract(self, image: Image) -> Image:
        from controlnet_aux.processor import Processor

        processor_id = 'mediapipe_face'
        processor = Processor(processor_id)

        processed_image = processor(image, to_pil=True)

        return processed_image

    def __str__(self) -> str:
        return 'Extractor(mediapipe_face)'

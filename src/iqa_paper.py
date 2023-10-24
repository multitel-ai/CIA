import hydra
import matplotlib.pyplot as plt
import os
import re

from omegaconf import DictConfig
from pathlib import Path
from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel
from tqdm import tqdm
from typing import List, Optional, Tuple

from common import logger, find_common_prefix, find_common_suffix
import numpy as np
import seaborn as sns
import json

# In this file the approach to measure quality will be the extensive library
# IQA-Pytorch: https://github.com/chaofengc/IQA-PyTorch
# Read also the paper: https://arxiv.org/pdf/2208.14818.pdf

# There are basically two approaches to measure image quality
# - full reference: compare againts a real pristine image
# - no reference: compute metrics following a learned opinion

# Because images are generated there is no reference image to compare to. We
# will be using with the no-reference metrics

# Note that methods using here are agnostic to the content of the image, no
# subjective or conceptual score is given.
# Measures generated here only give an idea of how 'good looking' the images
# are.

# Methods used:
# - brisque: https://www.sciencedirect.com/science/article/abs/pii/S0730725X17301340
# - dbccn: https://arxiv.org/pdf/1907.02665v1.pdf
# - niqe: https://live.ece.utexas.edu/research/quality/nrqa.html


"""
dbcnn is good for: blur, contrast distortion, white and pink noise, dithering, over and under exposure
brisque is good for: distortion, luminance and blur
ilniqe: distortion, blur and compression distortion

Note that metrics have each different ranges: [0, 1], [0, +inf] and also sometimes less is better
and sometimes more is better, it would be a mistake to try to rescale or invert them.
It is better to treat each separately.

There is file created in data/iqa/<cn_use>_iqa.json with the following structure:

{
    "image_paths": [
        "data/generated/controlnet_segmentation/000000000474_1.png",
        "data/generated/controlnet_segmentation/000000000474_2.png",
        "data/generated/controlnet_segmentation/000000000474_3.png",
        "data/generated/controlnet_segmentation/000000000474_4.png",
        "data/generated/controlnet_segmentation/000000000474_5.png"
    ],
    "name": "controlnet_segmentation",
    "brisque": [
        20.71453857421875,
        11.63690185546875,
        17.65740966796875,
        5.10711669921875,
        32.71502685546875
    ],
    "dbcnn": [
        0.7001792192459106,
        0.6730189323425293,
        0.5987531542778015,
        0.5892908573150635,
        0.5235083699226379
    ],
    "ilniqe": [
        27.35899873000946,
        34.540625520909074,
        26.03838433381286,
        25.595288318528816,
        34.6185153006446
    ]
}
"""


def measure_several_images(metric: InferenceModel,
                           image_paths: List[str],
                           ref_image_paths: Optional[List[str]] = None
                           ) -> Tuple[float, float]:
    number_of_images = len(image_paths)
    scores = []
    avg_score = 0

    for i, image_path in enumerate(tqdm(image_paths, unit='image')):
        ref_image_path = ref_image_paths and ref_image_paths[i]

        score = metric(image_path, ref_image_path)
        score = score.item()  # This should be adapted if using cpu as device,
                              # here because of cuda we get a 1-dim tensor

        scores.append(score)
        avg_score += score

    avg_score = avg_score / number_of_images
    return scores, avg_score


def is_generated_image(image_path: str) -> bool:
    # You should change the regex in this function to match whatever
    # naming convention you follow for you experience.
    regex = '^[0-9]+_[0-9]+.(jpg|png)'

    image_wo_path = os.path.basename(image_path)
    return re.match(regex, image_wo_path)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data']
    # keep track of what feature was used for generation too in the name
    base_path = os.path.join(*data_path['base']) if isinstance(data_path['base'], list) else data_path['base']
    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']

    IQA_PATH = Path(base_path) / 'iqa'
    IQA_PATH.mkdir(parents=True, exist_ok=True)
    file_json_iqa = IQA_PATH / f"{cfg['model']['cn_use']}_iqa.json"

    logger.info(f'Reading images from {GEN_DATA_PATH}')

    image_paths = [
        str(GEN_DATA_PATH / image_path)
        for image_path in os.listdir(str(GEN_DATA_PATH)) if is_generated_image(image_path)
    ]
    image_paths.sort()

    print(f"{GEN_DATA_PATH=} {base_path=}")

   # We are hard-coding the No-Reference methods for the moment.
   # See reasonment above.
    METRIC_MODE = 'NR'
    device = cfg['iqa']['device']
    # Hard coding needed metrics for the paper.
    metrics = ['brisque', 'dbcnn', 'ilniqe']

    dict_of_metrics = {}
    dict_of_metrics['image_paths'] = image_paths
    dict_of_metrics['name'] = f"{cfg['model']['cn_use']}"

    logger.info(f'Using a {METRIC_MODE} approach, metrics: {metrics} and device: {device}')

    for metric_name in metrics:
        iqa_model = create_metric(metric_name, device=device, metric_mode=METRIC_MODE)
        scores, _ = measure_several_images(iqa_model, image_paths)
        dict_of_metrics[metric_name] = scores

    logger.info(f"Writing metrics score to file {file_json_iqa}")

    json_object = json.dumps(dict_of_metrics, indent=4)
    with open(file_json_iqa, "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    main()

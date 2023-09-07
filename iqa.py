import hydra
from omegaconf import DictConfig, OmegaConf

from typing import List, Optional
import re
import glob
import os
from pathlib import Path
from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel
from tqdm import tqdm
from PIL import Image

# In this file the approach to measure quality will be the extensive library
# IQA-Pytorch: https://github.com/chaofengc/IQA-PyTorch
# Read also the paper: https://arxiv.org/pdf/2208.14818.pdf

# There are basically two approaches to measure image quality
# - full reference: compare againts a real pristine image
# - no reference: compute metrics following a learned opinion

# Because images are generated there is no reference image to compare to. We
# will be using with the no-reference metrics


# Some No-Reference methods:
# - brisque: Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE).
#   A BRISQUE model is trained on a database of images with known distortions,
#   and BRISQUE is limited to evaluating the quality of images with the same
#   type of distortion. BRISQUE is opinion-aware, which means subjective quality
#   scores accompany the training images.
# - niqe: Natural Image Quality Evaluator (NIQE). Although a NIQE model is
#   trained on a database of pristine images, NIQE can measure the quality of
#   images with arbitrary distortion. NIQE is opinion-unaware, and does not use
#   subjective quality scores. The tradeoff is that the NIQE score of an image
#   might not correlate as well as the BRISQUE score with human perception of
#   quality.
# - piqe: 	Perception based Image Quality Evaluator (PIQE). The PIQE algorithm
#   is opinion-unaware and unsupervised, which means it does not require a
#   trained model. PIQE can measure the quality of images with arbitrary
#   distortion and in most cases performs similar to NIQE. PIQE estimates
#   block-wise distortion and measures the local variance of perceptibly 
#   distorted blocks to compute the quality score.
# - biqi
# - cornia
# - hosa
# - tv
# - niqe
# - ilniqe
# - qac


def measure_several_images(metric: InferenceModel, image_paths: List[str], ref_image_paths: Optional[List[str]] = None):
    number_of_images = len(image_paths)
    scores = []
    avg_score = 0

    for i, image_path in enumerate(tqdm(image_paths, unit='image')):
        ref_image_path = ref_image_paths and ref_image_paths[i] 

        score = metric(image_path, ref_image_path)
        score = score.item()
        
        scores.append(score)
        avg_score += score

    avg_score = avg_score / number_of_images
    return scores, avg_score


def is_generated_image(image_path) -> bool:
    # You should change the regex in this function to match whatever
    # naming convention you follow for you experience.
    regex = '^[0-9]+_[0-9]+.(jpg|png)'
    
    image_wo_path = os.path.basename(image_path)
    return re.match(regex, image_wo_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

   # We are hard-coding the No-Reference methods for the moment.
   # See reasonment above.
    METRIC_MODE = 'NR'

    all_iqa_metrics = cfg['iqa']['metrics']
    metric = cfg['iqa']['current'].lower()
    metric = metric if metric in all_iqa_metrics else 'brisque'

    device = cfg['model']['device']

    iqa_model = create_metric(metric, device=device, metric_mode=METRIC_MODE)

    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data_path']
    REAL_DATA_PATH = Path(data_path['real'])
    # keep track of what feature was used for generation too in the name
    GEN_DATA_PATH =  Path(data_path['base']) / (f"{data_path['generated']}_{cfg['model']['cn_use']}")

    image_paths = [str(GEN_DATA_PATH / image_path) for image_path in os.listdir(str(GEN_DATA_PATH)) if is_generated_image(image_path)]

    scores, avg_scores = measure_several_images(iqa_model, image_paths)

    print(scores, avg_scores)

if __name__ == '__main__':
    main()

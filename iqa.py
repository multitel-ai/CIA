import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data_path']
    # keep track of what feature was used for generation too in the name
    GEN_DATA_PATH =  Path(data_path['base']) / (f"{data_path['generated']}_{cfg['model']['cn_use']}")

    image_paths = [
        str(GEN_DATA_PATH / image_path)
        for image_path in os.listdir(str(GEN_DATA_PATH)) if is_generated_image(image_path)
    ]
    image_paths.sort()
    image_names = [os.path.basename(image_path)[:4] for image_path in image_paths]

   # We are hard-coding the No-Reference methods for the moment.
   # See reasonment above.
    METRIC_MODE = 'NR'

    all_iqa_metrics = cfg['iqa']['metrics']
    metrics = [metric.lower() for metric in cfg['iqa']['current'] if metric.lower() in all_iqa_metrics]
    if not metrics:
        metrics = ['brisque']
    device = cfg['model']['device']

    overall_scores = {}
    for metric_name in tqdm(metrics):
        print('=' * 40)
        print(f'Measure using {metric_name} metric.')
        iqa_model = create_metric(metric_name, device=device, metric_mode=METRIC_MODE)
        scores, avg_score = measure_several_images(iqa_model, image_paths)
        overall_scores[metric_name] = scores, avg_score
    print('=' * 40)

    global_avg_score = 0
    for metric_name in overall_scores:
        scores, avg_score = overall_scores[metric_name]
        global_avg_score += avg_score
        plt.plot(image_names, scores, label = f'Avg score of {metric_name}: {avg_score}')
    global_avg_score = global_avg_score / len(metrics)

    plt.title(f'Dataset: {os.path.basename(str(GEN_DATA_PATH))}\nGlobal avg score: {global_avg_score}', loc='left')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

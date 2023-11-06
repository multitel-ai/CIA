import hydra
import os
import re
import json

from pathlib import Path
from omegaconf import DictConfig

from pyiqa import create_metric
from pyiqa.models.inference_model import InferenceModel
from tqdm import tqdm
from typing import List, Optional, Tuple

from common import logger, find_common_prefix, find_common_suffix


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


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
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
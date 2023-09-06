import cv2
import os
import numpy as np
from pathlib import Path
from extractors import OpenPose
from generators import SDCN
from PIL import Image
from diffusers.utils import load_image
import glob
import hydra
from omegaconf import DictConfig


@hydra.main(config_name='config')
def main(cfg: DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    BASE_PATH = Path(__file__).parent.resolve()
    DATA_PATH = BASE_PATH / "bank" / "data"

    # Loading COCO should be here somwhere
    formats = cfg.formats
    images = []
    
    for format in formats:
        images += [ 
            *glob.glob( str((DATA_PATH / "data" / "real").absolute()) + f'/*.{format}')
        ]

    images = [load_image(image_path) for image_path in images]

    # We should explore what prompts are better ? Let's write a prompts generator
    # number of prompts in the list = 
    ## either number of images
    ## or in case of 1 original image, it's the number of generations

    positive_prompt = [prompt + cfg.positive_prompt for prompt in cfg.positive_prompt]

    negative_prompt = cfg.negative_prompt * len(positive_prompt)
    # Specify the results path
    #results_path = DATA_PATH / "data" / "openpose1"

    openpose_dirs = sorted(glob.glob(str(DATA_PATH / "openpose*")))
    latest_experiment = max(int(Path(dir).name[8:]) for dir in openpose_dirs)
    new_experiment = latest_experiment + 1
    results_path = DATA_PATH / cfg.model_name + new_experiment 
    (results_path).mkdir(parents=True, exist_ok=True)

    # sdcn = SDCN("lllyasviel/sd-controlnet-canny")
    # sdcn = SDCN("lllyasviel/sd-controlnet-openpose")
    sdcn = SDCN(
        "runwayml/stable-diffusion-v1-5",
        "fusing/stable-diffusion-v1-5-controlnet-openpose",
        cfg.seed
    )

    extractions = []
    for image in images:
        # Feature Extraction
        extractions += [OpenPose().detect(np.array(image))]
        # extractions[-1].save(results_path / f"condition.png")

        
    i = 0
    for i, condition in enumerate(extractions): 
        # generate with stable diffusion
        output = sdcn.gen(condition, positive_prompt, negative_prompt)

        # save images
        for idx, img in enumerate(output.images):
            image_name = Path(images[idx]).stem
            img.save(results_path / f"{image_name}_{i%len(positive_prompt)}.png")
            i += 1

if __name__=="_main__":
    main()
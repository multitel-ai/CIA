import argparse
import cv2
import hydra
import glob
import os
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List

from diffusers.utils import load_image
import numpy as np
from pathlib import Path
from PIL import Image

from extractors import * 
from generators import SDCN


# Best clear way that I have to do this for the moment
extractors_dict = {
    'canny': Canny,
    'openpose': OpenPose,
    'fusing_openpose': OpenPose,
}

def find_model_name(name: str, l: List[Dict[str, str]]) -> str:
    for small_dict in l:
        if name in small_dict:
            return small_dict[name]
    return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data_path']
    REAL_DATA_PATH = Path(data_path['real'])
    # keep track of what feature was used for generation too in the name
    GEN_DATA_PATH =  Path(data_path['base']) / (f"{data_path['generated']}_{cfg['model']['cn_use']}")

    # Specify the results path
    (GEN_DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Loading COCO should be here somwhere
    formats = cfg['image_formats'] 
    images = []
    for format in formats:
        images += [ 
            *glob.glob(str(REAL_DATA_PATH.absolute()) + f'/*.{format}')
        ]
    images = [load_image(image_path) for image_path in images]

    # We should explore what prompts are better ? Let's write a prompts generator
    # number of prompts in the list = 
    ## either number of images
    ## or in case of 1 original image, it's the number of generations
    prompt = cfg['prompt']
    if isinstance(prompt['base'], str):
        positive_prompt = [prompt['base'] + ' ' + prompt['modifier'] + ' ' + prompt['quality']]
        negative_prompt = [''.join(prompt['negative_simple'])]
    else:
        positive_prompt = [
            pb + ' ' + prompt['modifier'] + ' ' + prompt['quality']
            for pb in prompt['base']
        ]
        negative_prompt = [''.join(prompt['negative_simple'])] * len(positive_prompt)

    # Specify the model and feature extractor. Be aware that ideally both extractor and
    # generator should be using the same feature.
    model_data = cfg['model']
    sd_model = model_data['sd']
    
    cn_model = find_model_name(model_data['cn_use'], model_data['cn'])
    cn_model = cn_model if cn_model is not None else 'fusing/stable-diffusion-v1-5-controlnet-openpose'
    
    seed = model_data['seed']
    device = model_data['device']

    generator = SDCN(sd_model, cn_model, seed, device=device)
    extractor = extractors_dict[
        model_data['cn_use'] if model_data['cn_use'] in extractors_dict else 'canny'
    ]()

    # Generate from each image several synthetic images following the different prompts.
    for i, image in enumerate(images):
        # Copy the original image to the same directory to ease the quality testing after.
        image.save(GEN_DATA_PATH / f'b_{i+1}.png')

         # Feature extraction, save also the features.
        feature = extractor.extract(image)
        feature.save(GEN_DATA_PATH / f"f_{i+1}.png")
        
        # Generate with stable diffusion
        output = generator.gen(feature, positive_prompt, negative_prompt)

        # save images
        for j, img in enumerate(output.images):
            img.save(GEN_DATA_PATH / f'{i+1}_{j+1}.png')


if __name__ == '__main__':
    main()

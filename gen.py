import cv2
import os
import numpy as np
from pathlib import Path
from extractors import OpenPose
from generators import SDCN
from PIL import Image
from diffusers.utils import load_image
import glob

SDCN_OPTIONS = {
    'openpose': [
        {
            'sd': 'runwayml/stable-diffusion-v1-5',
            'cn': 'lllyasviel/sd-controlnet-openpose',
        },
        {
            'sd': 'runwayml/stable-diffusion-v1-5',
            'cn': 'frankjoshua/control_v11p_sd15_openpose',
        },
        {
            'sd': 'stabilityai/stable-diffusion-2-1',
            'cn': 'thibaud/controlnet-sd21-openposev2-diffusers',
        },
        {
            'sd': 'stabilityai/stable-diffusion-2-1',
            'cn': 'thibaud/controlnet-sd21-openpose-diffusers',
        },
    ],
    'canny': [
        {
            'sd': 'runwayml/stable-diffusion-v1-5',
            'cn': 'lllyasviel/sd-controlnet-canny',
        },
    ],   
}

EXTRACTOR_TO_USE = 'openpose'
MODEL_TO_USE = 1

# BASE PATHS, please used these when specifying paths
BASE_PATH = Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH / "bank"

# Loading COCO should be here somwhere
formats = ['jpg', 'jpeg', 'png']
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
positive_prompt = ["Sandra Oh body", "Kim Kardashian body", "rihanna ", "taylor swift"]
positive_prompt = [prompt + " wearing jeans and a shirt, smiling, with a realistic face, and hands clapping" for prompt in positive_prompt]

negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, cartoon, unrealistic"] * len(positive_prompt)

# Specify the results path
results_path = DATA_PATH / "data" / (EXTRACTOR_TO_USE + str(MODEL_TO_USE))
(results_path).mkdir(parents=True, exist_ok=True)

sdcn = SDCN(
    SDCN_OPTIONS[EXTRACTOR_TO_USE][MODEL_TO_USE]['sd'],
    SDCN_OPTIONS[EXTRACTOR_TO_USE][MODEL_TO_USE]['cn'],
    1
)


extractions = []
for image in images:
    # Feature Extraction
    extractions += [OpenPose().detect(np.array(image))]
    # extractions[-1].save(results_path / f"condition.png")
    
i = 0
for i, condition in enumerate(extractions[1:2]): 
    # generate with stable diffusion
    output = sdcn.gen(condition, positive_prompt, negative_prompt)

    # save images
    for _, img in enumerate(output.images):
        img.save(results_path / f"{i}.png")
        i += 1

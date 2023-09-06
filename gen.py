import cv2
import glob
import os

from diffusers.utils import load_image
import numpy as np
from pathlib import Path
from PIL import Image

from extractors import OpenPose
from generators import SDCN


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
positive_prompt = [
    prompt + " wearing jeans and a shirt, smiling, with a realistic face, and hands clapping" 
    for prompt in positive_prompt
]

negative_prompt = [
    "monochrome, lowres, bad anatomy, worst quality, low quality, cartoon, unrealistic, bad proportion, no umbrella,"
    "distortion, bad quality, lowres, cropped,worst quality, bad focus, blurry, ad compression, bad artifact,"
    "bad pixel, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4),"
    " close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated,"
    "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated,"
    "bad anatomy, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs,"
    "extra arms, extra legs, fused fingers, too many fingers, long neck, no cars, no people, illustration, painting,"
    "drawing, art, sketch, cartoon, anime, deformation, distorsion",
] * len(positive_prompt)

# Specify the results path
results_path = DATA_PATH / "data" / "openpose1"
(results_path).mkdir(parents=True, exist_ok=True)

# sdcn = SDCN("lllyasviel/sd-controlnet-canny")
# sdcn = SDCN("lllyasviel/sd-controlnet-openpose")
sdcn = SDCN(
    "runwayml/stable-diffusion-v1-5",
    "fusing/stable-diffusion-v1-5-controlnet-openpose",
    2
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
    for _, img in enumerate(output.images):
        img.save(results_path / f"{i}.png")
        i += 1

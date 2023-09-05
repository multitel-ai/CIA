import torch
from typing import List
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler


class SDCN:
    def __init__(self, sd_model: str, control_model: str, seed: int):
        self.seed = seed
        
        self.controlnet = ControlNetModel.from_pretrained(
            control_model, 
            torch_dtype = torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, 
            controlnet = self.controlnet, 
            torch_dtype = torch.float16,
            safety_checker = None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def gen(self, condition: np.array | List[np.array], positive_prompts: List[str], negative_prompts: List[str]):
        generator = [torch.Generator(device="cpu").manual_seed(self.seed) for i in range(len(positive_prompts))]

        output = self.pipe(
            positive_prompts,
            condition,
            negative_prompt = negative_prompts,
            generator = generator,
            num_inference_steps = 20,
        )

        return output
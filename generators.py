import numpy as np
from typing import List
import random
import json

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

from logger import logger

class SDCN:
    def __init__(self, sd_model: str, control_model: str, seed: int, device='cpu',  cn_extra_settings = {}):

        logger.info(f'Initializing SDCN with {sd_model} and {control_model}, seed ={seed}, device={device}')

        self.seed = seed
        self.device = device
        self.control_model = control_model

        self.controlnet = ControlNetModel.from_pretrained(
            self.control_model,
            torch_dtype = torch.float16,
            **cn_extra_settings
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet = self.controlnet,
            torch_dtype = torch.float16,
            safety_checker = None,
        )

        # The default config seems to work best for the moment, we would need to tweak a lot to know
        # what to use.
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Docs of this scheduler: https://huggingface.co/docs/diffusers/main/en/api/schedulers/unipc
        # Example of scheduler with full parameters customization, does not seem to work fine...
        self.pipe.scheduler2 = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            trained_betas=None, # this is a list of floats or the equivalen 1-dim np.array
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1.0,
            predict_x0=True,
            solver_type="bh2",
            lower_order_final=True,
            disable_corrector=[],
            solver_p=None,
            # The arguments below are listed in the documentation but are not part of the current source code
            # use_karras_sigmas=False,
            # timestep_spacing="linspace",
            # steps_offset=0,
        )

        # Move the whole pipe to the designed device to avoid malformed cuda/cpu instructions
        self.pipe.to(device)

        # The line below is explained in https://huggingface.co/blog/controlnet but sometimes
        # it will throw an error later in the pipeline about having or not instructions for half
        # floats if you try to use 'cuda'.
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def __str__(self) -> str:
        return f'SDCN({self.control_model})'

    def gen(self,
            condition: np.array | List[np.array],
            positive_prompts: List[str],
            negative_prompts: List[str],
            quality: str = 30,
            guidance_scale: float = 7.0,
        ):
        generator = [
            torch.Generator(device=self.device).manual_seed(self.seed) for i in range(len(positive_prompts))
        ]

        output = self.pipe(
            positive_prompts,
            condition,
            negative_prompt = negative_prompts,
            generator = generator,
            num_inference_steps = quality,
            guidance_scale = guidance_scale,
        )

        return output

class Prompt:
    def __init__(self, VOCAB_TEMPLATE_PATH) -> None:

        self.vocabulary = json.load(open(VOCAB_TEMPLATE_PATH))['vocabulary']

        self.prompt_templates = json.load(open(VOCAB_TEMPLATE_PATH))['prompt_templates']

    def max_num_prompts(self, phrase: str) -> int:
        '''
        # Calculates the maximum number of prompts that can be generated for the given phrase
        Args: phrase; str; The basic prompt that needs to be changed based on options in vocabulary

        Returns: num_prompts; int; The maximum number of prompts that can be generated for the given phrase
        '''

        num_prompts = 1
        phrase_list = phrase.lower().split()
        for phrase_loc in range(len(phrase_list)):
            for vocabulary_class in self.vocabulary.keys():
                if phrase_list[phrase_loc] in self.vocabulary[vocabulary_class]:
                    num_prompts = num_prompts * len(self.vocabulary[vocabulary_class])
                    break

        return(num_prompts)

    def prompts(self, num_prompts: int, phrase: str) -> list:
        '''
        # Generates unique prompts in the format of the given phrase
        Args: num_prompts; int; Number of prompts that are required including 'phrase' (num_prompts given by max_num_prompts is used if this value is greater than maximum number of possible prompts)
              phrase; str; The basic prompt that needs to be changed based on options in vocabulary

        Returns: phrase_list; list; A list of unique prompts in the format of given phrase
        '''

        phrase_list = []
        phrase_list.extend([phrase.lower()])
        num_prompts = min(num_prompts, self.max_num_prompts(phrase))
        while(len(phrase_list) < num_prompts):
            new_phrase_list = phrase_list[0].lower().split()
            for phrase_loc in range(len(new_phrase_list)):
                for vocabulary_class in self.vocabulary.keys():
                    if new_phrase_list[phrase_loc] in self.vocabulary[vocabulary_class]:
                        new_phrase_list[phrase_loc] = random.choice(self.vocabulary[vocabulary_class])
                        break
            new_phrase = ' '.join(new_phrase_list)
            if new_phrase not in phrase_list:
                phrase_list.extend([new_phrase])

        return phrase_list

    def max_template_prompts(self) -> int:
        ''''
        # Counts the maximum number of template prompts that can be generated with the exisitng prompt_templates
        Args: None
        Returns: counter; int; maximum number of prompts that can be generated from prompt_templates
        '''

        counter = 0
        for phrase in self.phrase_templates:
            phrase_counter = 1
            phrase_list = phrase.split()
            for phrase_word in phrase_list:
                vocab_counter = 0
                if 'color' in phrase_word:
                    vocab_counter = len(self.vocabulary['color'])
                elif 'opt' in phrase_word:
                    vocab_counter = len(self.vocabulary[phrase_word[4:]])
                if vocab_counter>0:
                    phrase_counter = phrase_counter*vocab_counter
                    continue
            counter = counter + phrase_counter

        return counter

    def template_prompts(self, num_prompts: int) -> list:
        '''
        # Generates unique prompts from the template prompts
        Args: num_prompts; int; number of prompts that are required
        Returns: phrases; list; list of prompts
        '''

        phrases = []
        num_phrases = min(num_prompts, self.max_template_prompts())

        while(len(phrases) < num_phrases):
            phrase = random.choice(self.prompt_templates)
            color_count = phrase.count('opt_color')
            new_phrase = [phrase
                          .replace('opt_gender',random.choice(self.vocabulary['gender']))
                          .replace('opt_age',random.choice(self.vocabulary['age']))
                          .replace('opt_size',random.choice(self.vocabulary['size']))
                          .replace('opt_height',random.choice(self.vocabulary['height']))
                          .replace('opt_clothes_top',random.choice(self.vocabulary['clothes_top']))
                          .replace('opt_clothes_bottom',random.choice(self.vocabulary['clothes_bottom']))
                          .replace('opt_accessories',random.choice(self.vocabulary['accessories']))
                          .replace('opt_ground',random.choice(self.vocabulary['ground']))
                          .replace('opt_background',random.choice(self.vocabulary['background']))]

            for c in range(color_count+1):
                new_phrase = [new_phrase[0]
                            .replace(f'opt_color{c}',random.choice(self.vocabulary['color']))]

            if new_phrase[0] not in phrases:
                phrases.extend(new_phrase)

        return(phrases)

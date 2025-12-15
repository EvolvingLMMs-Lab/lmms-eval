import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import (
    Accelerator,
)
from accelerate.utils import send_to_device
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages

try:
    from lmms_engine.datasets.processor import BagelDataProcessor, ProcessorConfig
    from lmms_engine.models.bagel import Bagel
    from lmms_engine.models.bagel.inferencer import InterleaveInferencer
except Exception as e:
    eval_logger.error(f"Failed to import Bagel dependencies. {e}")
    eval_logger.error("Please install lmms-engine https://github.com/EvolvingLMMs-Lab/lmms-engine to use lmms-engine's bagel model")


@register_model("bagel_lmms_engine")
class BagelLmmsEngine(lmms):
    is_simple = False
    """
    Bagel LMMs Engine
    Supports text-to-image generation with optional thinking process

    Example usage:
    accelerate launch -m lmms_eval \
        --model bagel_lmms_engine \
        --model_args pretrained=lmms-lab/BAGEL-7B-MoT-ver.LE \
        --tasks ueval \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        output_image_dir: Optional[str] = None,
        show_thinking: bool = False,
        cfg_text_scale: float = 4.0,
        cfg_interval: float = 0.4,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        max_think_token_n: int = 1024,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        seed: int = 0,
        image_ratio: str = "1:1",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.show_thinking = show_thinking
        self.continual_mode = continual_mode

        # Generation hyperparameters
        self.cfg_text_scale = cfg_text_scale
        self.cfg_interval = cfg_interval
        self.timestep_shift = timestep_shift
        self.num_timesteps = num_timesteps
        self.cfg_renorm_min = cfg_renorm_min
        self.cfg_renorm_type = cfg_renorm_type
        self.max_think_token_n = max_think_token_n
        self.do_sample = do_sample
        self.text_temperature = text_temperature
        self.seed = seed
        self.image_ratio = image_ratio

        # Set image shapes based on ratio
        if image_ratio == "1:1":
            self.image_shapes = (1024, 1024)
        elif image_ratio == "4:3":
            self.image_shapes = (768, 1024)
        elif image_ratio == "3:4":
            self.image_shapes = (1024, 768)
        elif image_ratio == "16:9":
            self.image_shapes = (576, 1024)
        elif image_ratio == "9:16":
            self.image_shapes = (1024, 576)
        else:
            eval_logger.warning(f"Unknown image ratio {image_ratio}, using 1:1")
            self.image_shapes = (1024, 1024)

        if output_image_dir is None:
            self.output_image_dir = os.path.join(self.response_persistent_folder, "bagel_generated_images")
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading Bagel model from {pretrained}")
        self._load_model()

        eval_logger.info("Bagel model initialized successfully")

    def _load_model(self):
        """Load Bagel model components"""
        model_path = self.pretrained

        # Create model config
        self.config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=self.device_map)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor_config = ProcessorConfig(processor_name=model_path, processor_type="bagel")
        self.data_processor = BagelDataProcessor(processor_config)
        self.data_processor.build()
        tokenizer, new_token_ids, _ = self.data_processor.add_special_tokens(tokenizer)
        self._patch_prepare_methods(model, self._device)

        # Create transforms
        vae_transform = self.data_processor.vae_image_transform
        vit_transform = self.data_processor.vit_image_transform

        # Create inferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=model.vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        self._model = model
        self._tokenizer = tokenizer

    def _patch_prepare_methods(self, model, device):
        """Patch model's prepare_* methods to auto-move output tensors to device."""
        methods = ["prepare_prompts", "prepare_vae_images", "prepare_vit_images", "prepare_vae_latent", "prepare_vae_latent_cfg", "prepare_start_tokens"]

        for name in methods:
            if hasattr(model, name):
                orig_fn = getattr(model, name)
                setattr(model, name, lambda *a, fn=orig_fn, **kw: send_to_device(fn(*a, **kw), device))

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        if seed > 0:
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def generate_text_and_image(self, prompt: str, image: Image.Image, doc_id: str, task: str) -> Tuple[str, List[str]]:
        """
        Generate text and image from prompt

        Args:
            prompt: Input text prompt
            image: Input image
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        # Prepare inference hyperparameters
        inference_hyper = {
            "max_think_token_n": self.max_think_token_n if self.show_thinking else 1024,
            "do_sample": self.do_sample if self.show_thinking else False,
            "text_temperature": self.text_temperature if self.show_thinking else 0.3,
            "cfg_text_scale": self.cfg_text_scale,
            "cfg_interval": [self.cfg_interval, 1.0],
            "timestep_shift": self.timestep_shift,
            "num_timesteps": self.num_timesteps,
            "cfg_renorm_min": self.cfg_renorm_min,
            "cfg_renorm_type": self.cfg_renorm_type,
            "image_shapes": self.image_shapes,
            "enable_sde": False,  # Always disable SDE for eval
        }

        # Generate
        result = self.inferencer(text=prompt, think=self.show_thinking, image=image, **inference_hyper)

        # Extract text
        output_text = result.get("text", "")

        # Save image
        output_images = []
        if "image" in result and result["image"] is not None:
            image = result["image"]
            if isinstance(image, torch.Tensor):
                # Return PIL Image for compatibility
                image_pil = image.permute(1, 2, 0) * 255  # (H, W, C)
                image_pil = Image.fromarray(image_pil.to(torch.uint8).cpu().numpy())
                image = image_pil

            task_dir = os.path.join(self.output_image_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            safe_filename = f"{task}_{doc_id}.png"
            image_path = os.path.join(task_dir, safe_filename)
            image.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved image: {image_path}")

        return output_text, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for request in requests:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = request.arguments
            # Generate
            prompt = ctx
            chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
            chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
            images, videos, _ = chat_messages.extract_media()
            if len(images) == 0:
                image = None
            else:
                image = images[0]
            output_text, output_images = self.generate_text_and_image(prompt, image, str(doc_id), task)

            # Format output
            formatted_output = self.format_output(output_text, output_images)
            res.append(formatted_output)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError("Bagel is a generation model and does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")

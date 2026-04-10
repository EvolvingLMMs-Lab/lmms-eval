# coding=utf-8
# Copyright 2025 Gen-Verse.
#
# Licensed under the MIT License

import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add MMaDA repository to Python path
# Expected: lmms-eval/MMaDA/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
mmada_path = os.path.join(str(wd), "MMaDA")
if os.path.exists(mmada_path):
    sys.path.insert(0, mmada_path)
    eval_logger.info(f"Added MMaDA path to sys.path: {mmada_path}")
else:
    eval_logger.warning(
        f"MMaDA repository not found at {mmada_path}. "
        f"Please clone it: cd {wd} && git clone https://github.com/Gen-Verse/MMaDA.git"
    )


@register_model("mmada")
class MMaDA(lmms):
    """
    MMaDA: Multimodal Large Diffusion Language Model

    A unified diffusion foundation model supporting:
    - Text generation (semi-autoregressive)
    - Multimodal understanding
    - Text-to-image generation (non-autoregressive diffusion)
    - Image captioning
    - Complex reasoning tasks

    Model: Gen-Verse/MMaDA-8B-MixCoT (default)
    Paper: https://arxiv.org/abs/2505.15809
    GitHub: https://github.com/Gen-Verse/MMaDA

    Example usage for understanding:
    python -m lmms_eval \\
        --model mmada \\
        --model_args pretrained=Gen-Verse/MMaDA-8B-MixCoT,mode=understanding \\
        --tasks mmbench \\
        --batch_size 1 \\
        --output_path ./logs/

    Example usage for generation:
    python -m lmms_eval \\
        --model mmada \\
        --model_args pretrained=Gen-Verse/MMaDA-8B-MixCoT,mode=generation \\
        --tasks geneval \\
        --batch_size 1 \\
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "Gen-Verse/MMaDA-8B-MixCoT",
        mode: str = "understanding",
        weight_type: str = "bfloat16",
        output_image_dir: Optional[str] = None,
        guidance_scale: float = 3.5,
        generation_timesteps: int = 15,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        seed: int = 0,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.mode = mode
        self.pretrained = pretrained
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.guidance_scale = guidance_scale
        self.generation_timesteps = generation_timesteps
        self.seed = seed
        self.continual_mode = continual_mode

        # Set weight type
        if weight_type == "bfloat16":
            self.weight_type = torch.bfloat16
        elif weight_type == "float16":
            self.weight_type = torch.float16
        else:
            self.weight_type = torch.float32

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/mmada_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "mmada_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "mmada_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading MMaDA model from {pretrained}")
        self._load_model()
        eval_logger.info("MMaDA model initialized successfully")

    def _load_model(self):
        """Load MMaDA model components"""
        try:
            from transformers import AutoTokenizer
            from models import MAGVITv2, MMadaModelLM
            from training.prompting_utils import UniversalPrompting
            from training.utils import image_transform, image_transform_squash

            # Use accelerator's device for proper distributed inference
            self.device = self.accelerator.device
            eval_logger.info(
                f"Using device: {self.device} "
                f"(rank {self._rank}/{self._world_size})"
            )

            # Load tokenizer
            eval_logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained, padding_side="left", trust_remote_code=True
            )

            # Setup universal prompting
            self.uni_prompting = UniversalPrompting(
                self.tokenizer,
                max_text_len=512,  # Default max sequence length
                special_tokens=(
                    "<|soi|>",
                    "<|eoi|>",
                    "<|sov|>",
                    "<|eov|>",
                    "<|t2i|>",
                    "<|mmu|>",
                    "<|t2v|>",
                    "<|v2v|>",
                    "<|lvg|>",
                ),
                ignore_id=-100,
                cond_dropout_prob=0.0,
                use_reserved_token=True,
            )

            # Load VQ model (MAGVITv2)
            eval_logger.info("Loading VQ model...")
            self.vq_model = MAGVITv2.from_pretrained("showlab/magvitv2").to(
                self.device
            )
            self.vq_model.eval()
            self.vq_model.requires_grad_(False)

            # Load MMaDA model
            eval_logger.info("Loading MMaDA model...")
            self.model = MMadaModelLM.from_pretrained(
                self.pretrained,
                trust_remote_code=True,
                torch_dtype=self.weight_type,
            ).to(self.device)
            self.model.eval()

            # Store image transform functions
            self.image_transform = image_transform
            self.image_transform_squash = image_transform_squash

            eval_logger.info("Model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import MMaDA dependencies. "
                f"Please ensure:\n"
                f"  1. MMaDA repository is cloned at lmms-eval root\n"
                f"  2. Required dependencies are installed\n"
                f"Error: {e}"
            )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

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

    def understand_image(self, prompt: str, image: Image.Image, doc_id: str) -> str:
        """
        Understand image and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        try:
            # Transform image to tensor
            # Use squash transform for certain datasets, regular transform otherwise
            if any(
                tag in str(doc_id)
                for tag in ["ai2d", "clevr", "docvqa", "geo", "llava"]
            ):
                image_tensor = self.image_transform_squash(
                    image, resolution=512
                ).to(self.device)
            else:
                image_tensor = self.image_transform(
                    image, resolution=512
                ).to(self.device)

            image_tensor = image_tensor.unsqueeze(0)

            # Get image tokens from VQ model
            image_tokens = self.vq_model.get_code(image_tensor) + len(
                self.uni_prompting.text_tokenizer
            )

            # Prepare text input using chat template
            messages = [{"role": "user", "content": prompt}]
            text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

            # Construct input_ids with special tokens
            batch_size = image_tokens.shape[0]
            input_ids = torch.cat(
                [
                    (
                        torch.ones(batch_size, 1)
                        * self.uni_prompting.sptids_dict["<|mmu|>"]
                    ).to(self.device),
                    (
                        torch.ones(batch_size, 1)
                        * self.uni_prompting.sptids_dict["<|soi|>"]
                    ).to(self.device),
                    image_tokens,
                    (
                        torch.ones(batch_size, 1)
                        * self.uni_prompting.sptids_dict["<|eoi|>"]
                    ).to(self.device),
                    text_token_ids,
                ],
                dim=1,
            ).long()

            # Generate response
            # For understanding tasks, use optimized step count
            # Balance between speed and quality
            gen_max_tokens = self.max_new_tokens
            gen_steps = 64  # Increased to 64 steps for better quality on complex tasks
            gen_block_length = min(gen_max_tokens, 128)  # Reasonable block size

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.autocast("cuda", dtype=self.weight_type):
                        output_ids = self.model.mmu_generate(
                            input_ids,
                            max_new_tokens=gen_max_tokens,
                            steps=gen_steps,
                            block_length=gen_block_length,
                        )
                else:
                    output_ids = self.model.mmu_generate(
                        input_ids,
                        max_new_tokens=gen_max_tokens,
                        steps=gen_steps,
                        block_length=gen_block_length,
                    )

            # Decode generated tokens
            generated_ids = output_ids[:, input_ids.shape[1] :]
            response_text = self.uni_prompting.text_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response_text

        except Exception as e:
            eval_logger.error(f"Error in understand_image for doc_id={doc_id}: {e}")
            return ""

    def generate_text_and_image(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate text and image from prompt

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        try:
            # Import generate function from MMaDA
            from generate import generate

            # Prepare text input using chat template
            messages = [{"role": "user", "content": prompt}]
            text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

            # Add t2i special token for text-to-image generation
            batch_size = text_token_ids.shape[0]
            input_ids = torch.cat(
                [
                    (
                        torch.ones(batch_size, 1)
                        * self.uni_prompting.sptids_dict["<|t2i|>"]
                    ).to(self.device),
                    text_token_ids,
                ],
                dim=1,
            ).long()

            # Generate image tokens using diffusion
            with torch.no_grad():
                # Calculate generation parameters
                # Image tokens are typically 256 or 1024 depending on resolution
                gen_length = 256  # For 16x16 image tokens
                steps = self.generation_timesteps
                block_length = gen_length  # Non-autoregressive for images

                output_ids = generate(
                    self.model,
                    input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=self.temperature,
                    cfg_scale=self.guidance_scale,
                    remasking="low_confidence",
                    mask_id=self.tokenizer.mask_token_id,
                )

                # Extract generated image tokens
                generated_ids = output_ids[:, input_ids.shape[1] :]

                # Decode image tokens to image
                # Subtract offset to get VQ codes
                image_codes = generated_ids - len(self.uni_prompting.text_tokenizer)

                # Reshape to 2D grid (assuming 16x16)
                h = w = 16
                image_codes = image_codes.reshape(batch_size, h, w)

                # Decode using VQ model
                generated_image = self.vq_model.decode_code(image_codes)

                # Convert to PIL Image and save
                generated_image = torch.clamp(
                    (generated_image + 1.0) / 2.0, min=0.0, max=1.0
                )
                generated_image = (
                    generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                )
                generated_image = generated_image.astype(np.uint8)
                pil_image = Image.fromarray(generated_image)

                # Save image
                image_filename = f"{task}_{doc_id}.png"
                image_path = os.path.join(self.output_image_dir, image_filename)
                pil_image.save(image_path)

                output_text = f"Generated image for prompt: {prompt}"
                output_images = [image_path]

            return output_text, output_images

        except Exception as e:
            eval_logger.error(
                f"Error in generate_text_and_image for doc_id={doc_id}: {e}"
            )
            return "", []

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="MMaDA Generating",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            prompt = contexts

            if self.mode == "understanding":
                # Image understanding mode
                if doc_to_visual is None:
                    eval_logger.warning(
                        f"No visual provided for understanding mode, doc_id={doc_id}"
                    )
                    res.append("")
                    pbar.update(1)
                    continue

                # Get visuals from doc_to_visual
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                if not visuals or len(visuals) == 0:
                    eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Use first image for understanding
                image = visuals[0]
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                elif isinstance(image, Image.Image):
                    # Ensure image is in RGB mode (handle RGBA, L, etc.)
                    image = image.convert("RGB")
                else:
                    eval_logger.warning(
                        f"Unsupported visual type: {type(image)} for doc_id={doc_id}"
                    )
                    res.append("")
                    pbar.update(1)
                    continue

                output_text = self.understand_image(prompt, image, str(doc_id))
                formatted_output = output_text

            else:
                # Image generation mode
                output_text, output_images = self.generate_text_and_image(
                    prompt, str(doc_id), task
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "MMaDA is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation for MMaDA"
        )

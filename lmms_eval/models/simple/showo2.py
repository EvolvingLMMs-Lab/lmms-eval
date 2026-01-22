# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Show-o2 path will be added lazily in _load_model to avoid import conflicts
# Go up 5 levels from showo2.py to get to jxlei directory (sibling of lmms-eval)
wd = Path(__file__).parent.parent.parent.parent.parent.resolve()
showo2_path = os.path.join(str(wd), "Show-o", "show-o2")


@register_model("showo2")
class Showo2(lmms):
    """
    Show-o2 Multimodal Model
    Supports both image understanding and text-to-image generation

    Modes:
        - "understanding": Visual understanding (image + text -> text)
        - "generation": Image generation (text -> image)

    Example usage for understanding:
    python -m lmms_eval \
        --model showo2 \
        --model_args pretrained=showlab/show-o2-7B,mode=understanding \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for generation:
    python -m lmms_eval \
        --model showo2 \
        --model_args pretrained=showlab/show-o2-7B,mode=generation \
        --tasks geneval \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "showlab/show-o2-7B",
        mode: str = "understanding",
        llm_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        vae_model_path: Optional[str] = None,
        resolution: int = 432,
        weight_type: str = "bfloat16",
        output_image_dir: Optional[str] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        max_new_tokens: int = 512,
        top_k: int = 1,
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
        self.llm_model_path = llm_model_path
        self.vae_model_path = vae_model_path
        self.resolution = resolution
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.continual_mode = continual_mode

        # Set weight type
        if weight_type == "bfloat16":
            self.weight_type = torch.bfloat16
        elif weight_type == "float16":
            self.weight_type = torch.float16
        else:
            self.weight_type = torch.float32

        # Import Show-o2 dependencies (deferred to avoid path conflicts)
        try:
            # Add Show-o2 path now, after all other imports are done
            if os.path.exists(showo2_path) and showo2_path not in sys.path:
                sys.path.insert(0, showo2_path)
                eval_logger.info(f"Added Show-o2 path to sys.path: {showo2_path}")

            from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
            from models.misc import get_text_tokenizer
            from transport import Sampler, create_transport
            from utils import path_to_llm_name, denorm

            # Import image_transform using importlib to avoid conflict with HF datasets
            import importlib.util
            datasets_utils_path = os.path.join(showo2_path, "datasets", "utils.py")
            spec = importlib.util.spec_from_file_location("showo2_datasets_utils", datasets_utils_path)
            showo2_datasets_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(showo2_datasets_utils)
            image_transform = showo2_datasets_utils.image_transform

            self.Showo2Qwen2_5 = Showo2Qwen2_5
            self.WanVAE = WanVAE
            self.omni_attn_mask_naive = omni_attn_mask_naive
            self.get_text_tokenizer = get_text_tokenizer
            self.image_transform = image_transform
            self.create_transport = create_transport
            self.Sampler = Sampler
            self.path_to_llm_name = path_to_llm_name
            self.denorm = denorm

        except ImportError as e:
            raise ImportError(
                f"Failed to import Show-o2 dependencies. "
                f"Please ensure Show-o/show-o2 directory exists at lmms-eval root.\n"
                f"Error: {e}"
            )

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/showo2_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "showo2_generated_images"
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
                self.response_persistent_folder, "showo2_response.json"
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
        eval_logger.info(f"Loading Show-o2 model from {pretrained}")
        self._load_model()
        eval_logger.info("Show-o2 model initialized successfully")

    def _load_model(self):
        """Load Show-o2 model components"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load VAE model
        if self.vae_model_path is None:
            # Try to find VAE in common locations
            possible_paths = [
                "/scratch/azureml/cr/j/efa7581894b5472d91a754c6d79cc125/exe/wd/jxlei/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                os.path.join(str(wd), "models", "Wan2.1-T2V-14B", "Wan2.1_VAE.pth"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.vae_model_path = path
                    break
            if self.vae_model_path is None:
                raise ValueError(
                    "VAE model path not found. Please specify vae_model_path."
                )

        self.vae_model = self.WanVAE(
            vae_pth=self.vae_model_path, dtype=self.weight_type, device=device
        )

        # Load tokenizer
        llm_name = self.path_to_llm_name.get(self.llm_model_path, "qwen2_5")
        self.text_tokenizer, self.showo_token_ids = self.get_text_tokenizer(
            self.llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=llm_name,
        )

        # Load Show-o2 model
        self.model = self.Showo2Qwen2_5.from_pretrained(
            self.pretrained, use_safetensors=False
        ).to(device)
        self.model.to(self.weight_type)
        self.model.eval()

        # Setup hyperparameters based on resolution
        self._setup_hyperparams()

        # Setup transport for generation mode
        if self.mode == "generation":
            self._setup_transport()

    def _setup_hyperparams(self):
        """Setup hyperparameters based on resolution"""
        # Calculate latent dimensions based on resolution
        # For 432x432: latent is 27x27, patch_size=2, so num_tokens = 729
        # For 1024x1024: latent is 64x64, patch_size=2, so num_tokens = 4096
        self.patch_size = 2
        self.image_latent_dim = 16

        if self.resolution == 432:
            self.latent_height = 27
            self.latent_width = 27
            self.num_t2i_image_tokens = 729
            self.num_mmu_image_tokens = 729
            self.max_seq_len = 1024
        elif self.resolution == 1024:
            self.latent_height = 64
            self.latent_width = 64
            self.num_t2i_image_tokens = 4096
            self.num_mmu_image_tokens = 4096
            self.max_seq_len = 8192
        else:
            # Calculate for custom resolution
            self.latent_height = self.resolution // 16
            self.latent_width = self.resolution // 16
            self.num_t2i_image_tokens = (
                self.latent_height // self.patch_size
            ) * (self.latent_width // self.patch_size)
            self.num_mmu_image_tokens = self.num_t2i_image_tokens
            self.max_seq_len = self.num_t2i_image_tokens + 512

        # Check if model has time embeddings
        self.add_time_embeds = hasattr(self.model, "time_embed")
        if self.add_time_embeds:
            self.num_t2i_image_tokens += 1
            self.num_mmu_image_tokens += 1

        self.max_text_len = self.max_seq_len - self.num_t2i_image_tokens - 4

        # Token IDs
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = self.showo_token_ids["bos_id"]
        self.eos_id = self.showo_token_ids["eos_id"]
        self.boi_id = self.showo_token_ids["boi_id"]
        self.eoi_id = self.showo_token_ids["eoi_id"]
        self.img_pad_id = self.showo_token_ids["img_pad_id"]

    def _setup_transport(self):
        """Setup transport for image generation"""
        self.transport = self.create_transport(
            path_type="Linear",
            prediction="velocity",
            loss_weight=None,
            train_eps=None,
            sample_eps=None,
            snr_type="lognorm",
            do_shift=True,
            seq_len=self.num_t2i_image_tokens,
        )
        self.sampler = self.Sampler(self.transport)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer(self):
        return self.text_tokenizer

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

        # Transform image
        image_tensor = self.image_transform(
            image.convert("RGB"), resolution=self.resolution
        ).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        # Encode image with VAE
        image_latents = (
            self.vae_model.sample(image_tensor.unsqueeze(2))
            .squeeze(2)
            .to(self.weight_type)
        )

        # Get image embeddings
        image_embeds_und = self.model.image_embedder_und(image_latents)
        image_embeds_gen = self.model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + self.model.position_embedding(
            self.model.image_position_ids
        )
        image_embeds_und = self.model.und_trans(image_embeds_und)["last_hidden_state"]
        image_embeds = self.model.fusion_proj(
            torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
        )

        # Prepare text tokens
        sys_prompt_ids = self.text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
        )["input_ids"]
        role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)[
            "input_ids"
        ]
        role_b = self.text_tokenizer(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

        input_ids = self.text_tokenizer(prompt, add_special_tokens=False).input_ids
        text_tokens_a = torch.tensor(
            [self.showo_token_ids["bos_id"]] + sys_prompt_ids + role_a
        ).to(self.device)[None, :]
        text_tokens_b = torch.tensor(
            [self.showo_token_ids["boi_id"], self.showo_token_ids["eoi_id"]]
            + input_ids
            + role_b
        ).to(self.device)[None, :]

        text_embeds_a = self.model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self.model.showo.model.embed_tokens(text_tokens_b)

        if self.add_time_embeds:
            time_embeds = self.model.time_embed(
                torch.Tensor([[1.0]]).to(self.device), text_embeds_a.dtype
            )
            if hasattr(self.model, "time_embed_proj"):
                time_embeds = self.model.time_embed_proj(time_embeds)
            input_embeds = torch.cat(
                [
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    time_embeds,
                    image_embeds,
                    text_embeds_b[:, 1:],
                ],
                dim=1,
            ).to(self.weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 2, self.num_mmu_image_tokens]
            )[None, None, :].to(self.device)
        else:
            input_embeds = torch.cat(
                [
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    image_embeds,
                    text_embeds_b[:, 1:],
                ],
                dim=1,
            ).to(self.weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 1, self.num_mmu_image_tokens]
            )[None, None, :].to(self.device)

        attention_mask = self.omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=self.device,
            inverted=True,
        ).to(input_embeds.dtype)

        # Generate response
        output_tokens = self.model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            eos_token=self.text_tokenizer.eos_token_id,
        )

        output_tokens = torch.stack(output_tokens).squeeze()[None]
        text = self.text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return text[0]

    def understand_two_images(
        self, prompt: str, image1: Image.Image, image2: Image.Image, doc_id: str
    ) -> str:
        """
        Understand two images and answer question

        Args:
            prompt: Input text prompt/question
            image1: First PIL Image (e.g., original image)
            image2: Second PIL Image (e.g., auxiliary image)
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        # Transform and encode both images
        images = []
        for img in [image1, image2]:
            img_tensor = self.image_transform(
                img.convert("RGB"), resolution=self.resolution
            ).to(self.device)
            images.append(img_tensor.unsqueeze(0))

        # Stack images for batch processing
        images_batch = torch.cat(images, dim=0)

        # Encode images with VAE
        image_latents = (
            self.vae_model.sample(images_batch.unsqueeze(2))
            .squeeze(2)
            .to(self.weight_type)
        )

        # Get embeddings for both images
        all_image_embeds = []
        for i in range(2):
            latent = image_latents[i : i + 1]
            image_embeds_und = self.model.image_embedder_und(latent)
            image_embeds_gen = self.model.image_embedder_gen(latent)
            image_embeds_und = image_embeds_und + self.model.position_embedding(
                self.model.image_position_ids
            )
            image_embeds_und = self.model.und_trans(image_embeds_und)[
                "last_hidden_state"
            ]
            image_embeds = self.model.fusion_proj(
                torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
            )
            all_image_embeds.append(image_embeds)

        # Prepare text tokens
        sys_prompt_ids = self.text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False
        )["input_ids"]
        role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)[
            "input_ids"
        ]
        role_b = self.text_tokenizer(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

        input_ids = self.text_tokenizer(prompt, add_special_tokens=False).input_ids
        text_tokens_a = torch.tensor(
            [self.showo_token_ids["bos_id"]] + sys_prompt_ids + role_a
        ).to(self.device)[None, :]

        # Insert two image markers
        text_tokens_b = torch.tensor(
            [self.showo_token_ids["boi_id"], self.showo_token_ids["eoi_id"]]
            + [self.showo_token_ids["boi_id"], self.showo_token_ids["eoi_id"]]
            + input_ids
            + role_b
        ).to(self.device)[None, :]

        text_embeds_a = self.model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = self.model.showo.model.embed_tokens(text_tokens_b)

        # Build input with two images
        if self.add_time_embeds:
            time_embeds1 = self.model.time_embed(
                torch.Tensor([[1.0]]).to(self.device), text_embeds_a.dtype
            )
            time_embeds2 = self.model.time_embed(
                torch.Tensor([[1.0]]).to(self.device), text_embeds_a.dtype
            )
            if hasattr(self.model, "time_embed_proj"):
                time_embeds1 = self.model.time_embed_proj(time_embeds1)
                time_embeds2 = self.model.time_embed_proj(time_embeds2)

            input_embeds = torch.cat(
                [
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    time_embeds1,
                    all_image_embeds[0],
                    text_embeds_b[:, 1:2],
                    time_embeds2,
                    all_image_embeds[1],
                    text_embeds_b[:, 2:],
                ],
                dim=1,
            ).to(self.weight_type)

            # Two modality positions
            pos1 = text_tokens_a.shape[1] + 2
            pos2 = pos1 + self.num_mmu_image_tokens + 2
            modality_positions = torch.tensor(
                [[pos1, self.num_mmu_image_tokens], [pos2, self.num_mmu_image_tokens]]
            )[None, :, :].to(self.device)
        else:
            input_embeds = torch.cat(
                [
                    text_embeds_a,
                    text_embeds_b[:, :1],
                    all_image_embeds[0],
                    text_embeds_b[:, 1:2],
                    all_image_embeds[1],
                    text_embeds_b[:, 2:],
                ],
                dim=1,
            ).to(self.weight_type)

            pos1 = text_tokens_a.shape[1] + 1
            pos2 = pos1 + self.num_mmu_image_tokens + 1
            modality_positions = torch.tensor(
                [[pos1, self.num_mmu_image_tokens], [pos2, self.num_mmu_image_tokens]]
            )[None, :, :].to(self.device)

        attention_mask = self.omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=self.device,
            inverted=True,
        ).to(input_embeds.dtype)

        # Generate response
        output_tokens = self.model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            eos_token=self.text_tokenizer.eos_token_id,
        )

        output_tokens = torch.stack(output_tokens).squeeze()[None]
        text = self.text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return text[0]

    def generate_image(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate image from text prompt

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (empty_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        from models.misc import prepare_gen_input

        # Prepare input
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = (
            prepare_gen_input(
                [prompt],
                self.text_tokenizer,
                self.num_t2i_image_tokens,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.max_text_len,
                self.device,
            )
        )

        # Initialize noise
        z = torch.randn(
            (
                1,
                self.image_latent_dim,
                self.latent_height * self.patch_size,
                self.latent_width * self.patch_size,
            )
        ).to(torch.bfloat16).to(self.device)

        if self.guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat(
                [batch_text_tokens, batch_text_tokens_null], dim=0
            )
            modality_positions = torch.cat(
                [batch_modality_positions, batch_modality_positions_null], dim=0
            )
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions

        block_mask = self.omni_attn_mask_naive(
            text_tokens.size(0),
            self.max_seq_len,
            modality_positions,
            self.device,
        ).to(self.weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=self.max_seq_len,
            guidance_scale=self.guidance_scale,
        )

        sample_fn = self.sampler.sample_ode(
            sampling_method="euler",
            num_steps=self.num_inference_steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            time_shifting_factor=3.0,
        )
        samples = sample_fn(z, self.model.t2i_generate, **model_kwargs)[-1]
        samples = torch.chunk(samples, 2)[0]

        # Decode with VAE
        samples = samples.unsqueeze(2)
        images = self.vae_model.batch_decode(samples)
        images = images.squeeze(2)

        # Convert to PIL images
        images = self.denorm(images)
        pil_images = [Image.fromarray(img) for img in images]

        # Save images
        output_images = []
        for i, pil_image in enumerate(pil_images):
            safe_filename = f"{task}_{doc_id}_{i}.png"
            image_path = os.path.join(self.output_image_dir, safe_filename)
            pil_image.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved image: {image_path}")

        return "", output_images

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
            total=len(requests), disable=(self.rank != 0), desc="Show-o2 Generating"
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
                        f"No image provided for understanding mode, doc_id={doc_id}"
                    )
                    res.append("")
                    pbar.update(1)
                    continue

                # Get image from doc_to_visual
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                if not visuals or len(visuals) == 0:
                    eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Use first image for understanding
                image = visuals[0]
                output_text = self.understand_image(prompt, image, str(doc_id))
                formatted_output = output_text

            else:
                # Image generation mode
                output_text, output_images = self.generate_image(
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
            "Show-o2 is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")

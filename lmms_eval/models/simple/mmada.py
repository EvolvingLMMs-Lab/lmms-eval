"""
MMaDA (Multimodal Diffusion Language Model) Integration

MMaDA is a unified diffusion foundation model for text generation, image generation,
and multimodal understanding.

Usage:
    python -m lmms_eval \
        --model mmada \
        --model_args pretrained=/scratch/models/MMaDA-8B-MixCoT \
        --tasks chartqa100 \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Modified to work in-place with original dtype to minimize memory usage.
    """
    if temperature == 0:
        return logits
    # Work directly with the original dtype to avoid memory overhead from conversion
    # Generate noise in the same dtype as logits
    noise = torch.rand_like(logits)
    # Compute gumbel noise: (-log(noise))^temperature
    gumbel_noise = (-torch.log(noise)) ** temperature
    # Return logits.exp() / gumbel_noise
    # Use in-place operations where possible to save memory
    result = logits.exp()
    result.div_(gumbel_noise)
    return result


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    attention_mask=None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    """
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    else:
        attention_bias = None
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    # Ensure steps is at least num_blocks so each block gets at least 1 step
    steps = max(num_blocks, steps)
    # Adjust steps to be divisible by num_blocks
    steps = (steps // num_blocks) * num_blocks
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length :] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits

            # Get argmax directly without adding noise if temperature is 0
            if temperature == 0:
                x0 = torch.argmax(logits, dim=-1)  # b, l
            else:
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                del logits_with_noise

            if remasking == "low_confidence":
                # Use original dtype to avoid memory overhead
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                del p
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Free logits memory
            del logits

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def image_transform(image, resolution=512, normalize=True):
    """Transform image for MMaDA"""
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


@register_model("mmada")
class MMaDA(lmms):
    """
    MMaDA: Multimodal Diffusion Language Model

    A unified diffusion foundation model for text generation, image generation,
    and multimodal understanding.
    """

    def __init__(
        self,
        pretrained: str = "/scratch/models/MMaDA-8B-MixCoT",
        vq_model_path: str = "/scratch/models/magvitv2",
        # Generation parameters
        max_new_tokens: int = 512,
        steps: int = 64,
        block_length: int = 64,
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        # Image processing
        image_resolution: int = 128,  # Reduced from 256 to 128 to save memory
        # Model loading
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: str = "cuda",
        device_map: str = "auto",  # Use "auto" for multi-GPU model parallel
        batch_size: int = 1,
        # Output
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.vq_model_path = vq_model_path
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.image_resolution = image_resolution
        self.device_str = device
        self.device_map = device_map
        self.batch_size_per_gpu = batch_size
        self.mask_id = 126336  # MMaDA's mask token ID

        # Setup output directory
        if output_dir is None:
            self.output_dir = "./logs/mmada"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Load model
        eval_logger.info(f"Loading MMaDA model from {pretrained}")
        self._load_model(load_in_4bit, load_in_8bit)

        eval_logger.info("MMaDA initialized successfully")

    def _load_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """Load MMaDA model and VQ model"""
        from transformers import AutoTokenizer

        # Add MMaDA models directory to Python path
        mmada_repo_path = "/scratch/models/MMaDA"
        if mmada_repo_path not in sys.path:
            sys.path.insert(0, mmada_repo_path)

        # Import MMaDA models
        from models import MMadaModelLM, MAGVITv2

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained, trust_remote_code=True)

        # Set chat template
        self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"

        # Setup special tokens with reserved IDs (from MMaDA's prompting_utils.py)
        self.special_tokens = {
            "<|soi|>": 126084,
            "<|eoi|>": 126085,
            "<|sov|>": 126086,
            "<|eov|>": 126087,
            "<|t2i|>": 126088,
            "<|mmu|>": 126089,
            "<|t2v|>": 126090,
            "<|v2v|>": 126091,
            "<|lvg|>": 126092,
        }

        dtype = torch.bfloat16
        if load_in_4bit or load_in_8bit:
            eval_logger.warning("4-bit/8-bit quantization not yet supported for MMaDA, loading in bfloat16")

        # Load VQ model on GPU (will use first available GPU)
        eval_logger.info(f"Loading VQ model from {self.vq_model_path}")
        vq_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vq_model = MAGVITv2.from_pretrained(
            self.vq_model_path,
            low_cpu_mem_usage=False
        ).to(vq_device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)
        eval_logger.info(f"VQ model loaded successfully on {vq_device}")

        # Load MMaDA model with device_map for multi-GPU
        eval_logger.info(f"Loading MMaDA model from {self.pretrained} with device_map={self.device_map}")

        if self.device_map == "auto":
            # Auto device map will distribute model across available GPUs
            self.model = MMadaModelLM.from_pretrained(
                self.pretrained,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
                max_memory={0: "38GB", 1: "38GB"}  # Reserve some memory on each GPU
            ).eval()
            eval_logger.info(f"Model loaded with automatic device mapping across GPUs")
        else:
            # Single device
            device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
            self.model = MMadaModelLM.from_pretrained(
                self.pretrained,
                trust_remote_code=True,
                torch_dtype=dtype
            ).to(device).eval()
            eval_logger.info(f"Model loaded on {device} with dtype {dtype}")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input):
        """Flatten nested lists"""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for a list of requests"""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            desc="Generating with MMaDA",
        )

        # Group requests by generation kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)

            task = task[0]
            split = split[0]

            # Get visuals
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            # Get generation kwargs
            gen_kwargs = all_gen_kwargs[0]

            # Process each context
            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Handle batch processing
            for i, context in enumerate(contexts):
                # Get images for this context
                # Note: visuals are already flattened, so we take all of them for this single context
                images = []
                for visual in visuals:
                    if isinstance(visual, str):
                        visual = Image.open(visual).convert("RGB")
                    elif isinstance(visual, Image.Image):
                        visual = visual.convert("RGB")
                    images.append(visual)

                # Generate response
                if len(images) > 0:
                    # Multimodal generation
                    response = self._generate_multimodal(context, images, gen_kwargs)
                else:
                    # Text-only generation
                    response = self._generate_text_only(context, gen_kwargs)

                res.append(response)
                pbar.update(1)

        pbar.close()
        return res

    def _generate_text_only(self, context: str, gen_kwargs: dict) -> str:
        """Generate text-only response"""
        # Prepare prompt
        messages = [{"role": "user", "content": context}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(text=prompt, return_tensors="pt", padding=True, padding_side="left")["input_ids"]
        input_ids = input_ids.to(self.model.device)

        # Get generation parameters
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        steps = min(self.steps, max_new_tokens)
        block_length = min(self.block_length, max_new_tokens)

        # Generate
        output = generate(
            self.model,
            input_ids,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            remasking=self.remasking,
            mask_id=self.mask_id,
        )

        # Decode
        generated_text = self.tokenizer.batch_decode(output[:, input_ids.shape[1] :], skip_special_tokens=True)[0]
        return generated_text

    def _generate_multimodal(self, context: str, images: List[Image.Image], gen_kwargs: dict) -> str:
        """Generate multimodal response"""
        # Get the device of the first model parameter
        device = next(self.model.parameters()).device

        # Process images with VQ model on GPU
        image_tokens_list = []
        for img in images:
            img_tensor = image_transform(img, resolution=self.image_resolution, normalize=True)
            img_tensor = img_tensor.unsqueeze(0).to(self.vq_model.device)

            # Get image tokens from VQ model
            with torch.no_grad():
                img_tokens = self.vq_model.get_code(img_tensor) + len(self.tokenizer)

            # Move tokens to main model's device
            img_tokens = img_tokens.to(device)
            image_tokens_list.append(img_tokens)

        # Concatenate all image tokens
        if len(image_tokens_list) > 0:
            image_tokens = torch.cat(image_tokens_list, dim=1)
        else:
            image_tokens = None

        # Prepare text
        messages = [{"role": "user", "content": context}]
        text_token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Assemble input: <|mmu|> <|soi|> image_tokens <|eoi|> text_tokens
        batch_size = 1
        input_ids_list = [
            torch.ones(batch_size, 1, dtype=torch.long, device=device) * self.special_tokens["<|mmu|>"],
        ]

        if image_tokens is not None:
            input_ids_list.append(torch.ones(batch_size, 1, dtype=torch.long, device=device) * self.special_tokens["<|soi|>"])
            input_ids_list.append(image_tokens)
            input_ids_list.append(torch.ones(batch_size, 1, dtype=torch.long, device=device) * self.special_tokens["<|eoi|>"])

        input_ids_list.append(text_token_ids)
        input_ids = torch.cat(input_ids_list, dim=1).long()

        # Get generation parameters
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        steps = min(self.steps, max_new_tokens)
        block_length = min(self.block_length, max_new_tokens)

        # Generate
        output = generate(
            self.model,
            input_ids,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            remasking=self.remasking,
            mask_id=self.mask_id,
        )

        # Decode
        generated_ids = output[:, input_ids.shape[1]:]
        eval_logger.debug(f"Generated IDs shape: {generated_ids.shape}")
        eval_logger.debug(f"Generated IDs (first 20): {generated_ids[0, :20].tolist()}")
        eval_logger.debug(f"Mask ID: {self.mask_id}")
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        eval_logger.debug(f"Generated text: '{generated_text}'")
        return generated_text

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Calculate log-likelihood for requests"""
        eval_logger.warning("loglikelihood not implemented for MMaDA, returning dummy values")
        return [(0.0, False) for _ in requests]

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Generate responses for multi-round conversations"""
        eval_logger.warning("generate_until_multi_round not implemented for MMaDA, using generate_until")
        return self.generate_until(requests)

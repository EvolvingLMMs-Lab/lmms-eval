import math
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_NON_GENERATE_KEYS = {
    "add_special_tokens",
    "downsample_ratio",
    "enable_thinking",
    "max_pixels",
    "min_pixels",
    "patch_size",
    "remove_think",
    "system_prompt",
    "until",
    "upscale",
}


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any) -> float:
    return float(value)


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: Union[int, float], factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: Union[int, float], factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _contrasting_background(image: Image.Image) -> Optional[Tuple[int, int, int]]:
    image_alpha = image.getchannel("A")
    if not image_alpha.getextrema()[0] == 0:
        return None

    pixels = image.getdata()
    non_transparent = [pixel[:3] for pixel in pixels if pixel[3] > 0]
    if not non_transparent:
        return None
    pixel_mean = sum(sum(pixel) for pixel in non_transparent) / (len(non_transparent) * 3)
    return (0, 0, 0) if pixel_mean > 382.5 else (255, 255, 255)


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        bg_color = _contrasting_background(image)
        if bg_color:
            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=image.split()[3])
            return background.convert("RGB")
    return image.convert("RGB")


def _load_image(image_input: Any) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.copy()
    if isinstance(image_input, (str, Path, os.PathLike)):
        return Image.open(image_input)
    raise TypeError(f"Unsupported image type for neo_ov: {type(image_input)}")


def _preprocess_pixel_values(pixel_values: torch.Tensor, patch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flat_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )
    return flat_pixel_values, torch.tensor([[grid_h, grid_w]], device=pixel_values.device)


def _load_image_native(
    image_input: Any,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = _ensure_rgb(_load_image(image_input))
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    resized_height, resized_width = _smart_resize(
        image.height,
        image.width,
        factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    return _preprocess_pixel_values(transform(image).to(torch.float32), patch_size=patch_size)


@register_model("neo_ov")
class NeoOV(lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "Paranioar/NEO1_5-2B-SFT",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        patch_size: int = 16,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
        downsample_ratio: float = 0.5,
        system_prompt: Optional[str] = "Reason step by step.",
        max_new_tokens: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code, use_fast=False)
        self._model = AutoModel.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
        ).eval()

        self.batch_size_per_gpu = int(batch_size)
        self.patch_size = _as_int(patch_size)
        self.min_pixels = _as_int(min_pixels)
        self.max_pixels = _as_int(max_pixels)
        self.downsample_ratio = _as_float(downsample_ratio)
        self.system_prompt = system_prompt
        self.max_new_tokens = _as_int(max_new_tokens)
        self._config = self.model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 2048)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for neo_ov")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[GenerationResult]:
        raise NotImplementedError("Multi-round generation is not implemented for neo_ov")

    def _serialize_messages(self, messages: ChatMessages) -> Tuple[str, List[Any], Optional[str]]:
        prompt_parts = []
        images = []
        system_prompt = None

        for message in messages.messages:
            if message.role == "system":
                system_text = "\n".join(content.text for content in message.content if content.type == "text").strip()
                if system_text:
                    system_prompt = system_text
                continue

            for content in message.content:
                if content.type == "text":
                    prompt_parts.append(content.text)
                elif content.type == "image":
                    prompt_parts.append("<image>\n")
                    images.append(content.url)
                elif content.type in {"video", "audio"}:
                    raise NotImplementedError(f"neo_ov currently supports image inputs only, got {content.type}")

        return "".join(prompt_parts).strip(), images, system_prompt

    def _build_image_inputs(
        self,
        images: List[Any],
        gen_kwargs: dict,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not images:
            return None, None

        patch_size = _as_int(gen_kwargs.get("patch_size", self.patch_size))
        min_pixels = _as_int(gen_kwargs.get("min_pixels", self.min_pixels))
        max_pixels = _as_int(gen_kwargs.get("max_pixels", self.max_pixels))
        downsample_ratio = _as_float(gen_kwargs.get("downsample_ratio", self.downsample_ratio))
        upscale_value = gen_kwargs.get("upscale", False)
        upscale = str(upscale_value).lower() == "true" if isinstance(upscale_value, str) else bool(upscale_value)

        pixel_values_list = []
        grid_hw_list = []
        for image in images:
            pixel_values, grid_hw = _load_image_native(
                image,
                patch_size=patch_size,
                downsample_ratio=downsample_ratio,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                upscale=upscale,
            )
            target_device = "cuda" if self.device_map == "auto" else self.device
            pixel_values_list.append(pixel_values.to(target_device).to(torch.bfloat16))
            grid_hw_list.append(grid_hw.to(target_device))

        return torch.cat(pixel_values_list, dim=0), torch.cat(grid_hw_list, dim=0)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        generation_config = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "top_p": None,
        }
        generation_config.update({k: v for k, v in gen_kwargs.items() if k not in _NON_GENERATE_KEYS})

        if generation_config.get("do_sample") is False:
            if generation_config.get("temperature", None) == 0:
                generation_config["temperature"] = None
            if generation_config.get("top_p", None) == 1.0:
                generation_config["top_p"] = None
        return {k: v for k, v in generation_config.items() if v is not None}

    @staticmethod
    def _strip_thinking(response: str) -> str:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL | re.IGNORECASE)
        return response.strip()

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for request in requests:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = ChatMessages(**{"messages": doc_to_messages(doc)})
            prompt, images, message_system_prompt = self._serialize_messages(messages)
            request_system_prompt = gen_kwargs.get("system_prompt", message_system_prompt if message_system_prompt is not None else self.system_prompt)
            if request_system_prompt is not None:
                self.model.system_message = str(request_system_prompt).replace("\\n", "\n")

            pixel_values, grid_hw = self._build_image_inputs(images, gen_kwargs)
            generation_config = self._build_generation_config(gen_kwargs)

            start_time = time.time()
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
                question=prompt,
                generation_config=generation_config,
                verbose=self.rank == 0,
            )
            end_time = time.time()

            if gen_kwargs.get("remove_think", False):
                response = self._strip_thinking(response)

            output_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
            total_elapsed_time += end_time - start_time
            total_tokens += output_tokens
            res.append(GenerationResult(text=response, token_counts=TokenCounts(output_tokens=output_tokens)))
            self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), response)
            eval_logger.debug(f"Question: {prompt}")
            eval_logger.debug(f"Model Response: {response}")
            pbar.update(1)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res

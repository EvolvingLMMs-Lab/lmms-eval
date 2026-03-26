from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("penguinvl")
class PenguinVL(lmms):
    """
    Penguin-VL model wrapper for lmms-eval.
    """

    def __init__(
        self,
        pretrained: str = "tencent/Penguin-VL-8B",
        model_path: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
        add_system_prompt: bool = True,
        max_num_frames: int = 180,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 2048,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        max_length = kwargs.pop("max_length", 2048)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        pretrained = model_path or pretrained
        if max_frames is not None:
            max_num_frames = max_frames

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": utils.get_dtype(dtype),
            "device_map": self.device_map,
        }
        if enforce_eager:
            model_kwargs["attn_implementation"] = "eager"
        elif attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self._tokenizer = tokenizer

        self.system_prompt = self._resolve_system_prompt(system_prompt)
        self.add_generation_prompt = add_generation_prompt
        self.add_system_prompt = add_system_prompt
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_top_k = top_k
        self.default_repetition_penalty = repetition_penalty
        self.default_max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self._max_length = max_length
        self._config = self.model.config

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
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

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
        raise NotImplementedError("Loglikelihood is not implemented for PenguinVL")

    def _is_video_path(self, visual: Any) -> bool:
        return isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        array = tensor.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        return Image.fromarray(np.asarray(array).astype(np.uint8))

    def _normalize_clip_item(self, visual: Any) -> Any:
        if isinstance(visual, torch.Tensor):
            if visual.ndim == 4:
                return [self._tensor_to_pil(frame) for frame in visual]
            return self._tensor_to_pil(visual)
        if isinstance(visual, np.ndarray):
            if visual.ndim == 4:
                return [Image.fromarray(frame) for frame in visual]
            return Image.fromarray(visual)
        return visual

    def _normalize_visuals_to_clips(self, visuals: Any) -> list[list[Any]]:
        if visuals is None:
            return []

        if isinstance(visuals, (np.ndarray, torch.Tensor)) and getattr(visuals, "ndim", None) == 4:
            normalized = self._normalize_clip_item(visuals)
            return [normalized if isinstance(normalized, list) else [normalized]]

        if isinstance(visuals, (list, tuple)):
            if len(visuals) == 0:
                return []
            if isinstance(visuals[0], (list, tuple)):
                clips = []
                for clip in visuals:
                    normalized_clip = []
                    for item in clip:
                        normalized_item = self._normalize_clip_item(item)
                        if isinstance(normalized_item, list):
                            normalized_clip.extend(normalized_item)
                        else:
                            normalized_clip.append(normalized_item)
                    clips.append(normalized_clip)
                return clips

            clips = []
            for item in visuals:
                normalized_item = self._normalize_clip_item(item)
                if isinstance(normalized_item, list):
                    clips.append(normalized_item)
                else:
                    clips.append([normalized_item])
            return clips

        normalized = self._normalize_clip_item(visuals)
        return [normalized if isinstance(normalized, list) else [normalized]]

    def _build_visual_content(self, clip: list[Any], wrap_single_image_in_list: bool = False) -> dict[str, Any]:
        if len(clip) == 0:
            raise ValueError("Clip must not be empty.")

        if len(clip) == 1:
            visual = clip[0]
            if isinstance(visual, Image.Image):
                return {"type": "image", "image": [visual] if wrap_single_image_in_list else visual}
            if isinstance(visual, str):
                if self._is_video_path(visual):
                    return {
                        "type": "video",
                        "video": {"video_path": visual, "fps": self.fps, "max_frames": self.max_num_frames},
                    }
                return {"type": "image", "image": {"image_path": visual}}
            raise TypeError(f"Unsupported visual input type: {type(visual)}")

        if all(isinstance(frame, Image.Image) for frame in clip):
            return {"type": "video", "video": clip}

        raise TypeError(f"Unsupported clip input types: {[type(frame) for frame in clip]}")

    def _split_text_by_image_tokens(self, context: str) -> list[str]:
        return re.split(r"<image(?: \d+)?>", context)

    def _build_conversation(self, context: str, visuals: Any) -> list[dict[str, Any]]:
        conversation: list[dict[str, Any]] = []
        if self.system_prompt:
            conversation.append({"role": "system", "content": self.system_prompt})

        visual_clips = self._normalize_visuals_to_clips(visuals)
        wrap_single_image_in_list = len(visual_clips) > 1
        visual_contents = [self._build_visual_content(clip, wrap_single_image_in_list=wrap_single_image_in_list) for clip in visual_clips]
        if visual_contents:
            image_token_count = len(re.findall(r"<image(?: \d+)?>", context))
            if image_token_count == len(visual_contents):
                content = []
                text_parts = self._split_text_by_image_tokens(context)
                for idx, visual_content in enumerate(visual_contents):
                    if idx < len(text_parts) and text_parts[idx]:
                        content.append({"type": "text", "text": text_parts[idx]})
                    content.append(visual_content)
                if len(text_parts) > len(visual_contents) and text_parts[-1]:
                    content.append({"type": "text", "text": text_parts[-1]})
            else:
                cleaned_context = re.sub(r"<image(?: \d+)?>", "", context).strip()
                content = visual_contents + [{"type": "text", "text": cleaned_context}]
        else:
            content = context

        conversation.append({"role": "user", "content": content})
        return conversation

    def _move_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        target_device = "cuda" if self.device_map == "auto" else self.device
        moved_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(target_device)
                if key == "pixel_values":
                    value = value.to(torch.bfloat16)
            moved_inputs[key] = value
        return moved_inputs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visual_list = [fn(self.task_dict[t][s][i]) for fn, t, s, i in zip(doc_to_visual, task, split, doc_id)]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            current_gen_kwargs = {
                "max_new_tokens": self.default_max_new_tokens,
                "temperature": self.default_temperature,
                "top_p": self.default_top_p,
                "top_k": self.default_top_k,
                "num_beams": gen_kwargs.get("num_beams", 1),
                "repetition_penalty": self.default_repetition_penalty,
            }
            do_sample = current_gen_kwargs["temperature"] > 0

            for context, visuals in zip(contexts, visual_list):
                conversation = self._build_conversation(context, visuals)
                print("conversation,", conversation)
                inputs = self.processor(
                    conversation=conversation,
                    return_tensors="pt",
                    add_system_prompt=self.add_system_prompt,
                    add_generation_prompt=self.add_generation_prompt,
                )
                inputs = self._move_inputs_to_device(inputs)

                generate_kwargs = {
                    "do_sample": do_sample,
                    "num_beams": current_gen_kwargs["num_beams"],
                    "max_new_tokens": current_gen_kwargs["max_new_tokens"],
                    "repetition_penalty": current_gen_kwargs["repetition_penalty"],
                    "use_cache": self.use_cache,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                if current_gen_kwargs["temperature"] is not None:
                    generate_kwargs["temperature"] = current_gen_kwargs["temperature"]
                if current_gen_kwargs["top_p"] is not None:
                    generate_kwargs["top_p"] = current_gen_kwargs["top_p"]
                if current_gen_kwargs["top_k"] is not None:
                    generate_kwargs["top_k"] = current_gen_kwargs["top_k"]

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        **generate_kwargs,
                    )

                input_length = inputs["input_ids"].shape[1]
                if output_ids.shape[1] > input_length:
                    response_ids = output_ids[:, input_length:]
                else:
                    response_ids = output_ids

                answer = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
                for term in until:
                    if term:
                        answer = answer.split(term)[0]

                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

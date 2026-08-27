import os
import re
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Gemma 4 vision soft-token budgets
ALLOWED_MAX_SOFT_TOKENS = frozenset({70, 140, 280, 560, 1120})
DEFAULT_MAX_SOFT_TOKENS = 280
DEFAULT_MAX_FRAMES = 32
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg")


@register_model("gemma4")
class Gemma4(lmms):
    """
    Gemma 4 model.
    https://huggingface.co/google/gemma-4-31B-it
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-4-31B-it",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
        max_num_frames: int = DEFAULT_MAX_FRAMES,
        interleave_visuals: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if max_soft_tokens not in ALLOWED_MAX_SOFT_TOKENS:
            raise ValueError(f"max_soft_tokens must be one of {sorted(ALLOWED_MAX_SOFT_TOKENS)}, got {max_soft_tokens}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code, padding_side="left")
        self.processor.video_processor.sample_frames = self._sample_video_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self._model.config
        self._max_length = self._config.text_config.max_position_embeddings
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.max_soft_tokens = max_soft_tokens
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.model.eval()

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
        raise NotImplementedError("Not implemented for Gemma4.")

    def _encode_image_data_url(self, image: Image.Image) -> str:
        return encode_image_to_data_url(
            image,
            image_format="JPEG",
            mime_type="image/jpeg",
            convert_rgb=True,
            quality=85,
        )

    def _sample_video_frames(self, metadata, num_frames=None, **kwargs):
        num_frames = min(num_frames or self.max_num_frames, metadata.total_num_frames)
        return np.linspace(0, metadata.total_num_frames - 1, num_frames, dtype=int)

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
            visual_list = [visual_fn(self.task_dict[task_name][split_name][ids]) for visual_fn, ids, task_name, split_name in zip(doc_to_visual, doc_id, task, split)]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                visual_groups = []
                for visual in visual_list[i]:
                    visual_group = []
                    try:
                        if isinstance(visual, str) and visual.lower().endswith(VIDEO_EXTENSIONS):
                            if not os.path.exists(visual):
                                eval_logger.warning(f"Video file not found: {visual}")
                            else:
                                visual_group.append({"type": "video", "video": visual})
                        elif isinstance(visual, Image.Image):
                            visual_group.append({"type": "image", "image": self._encode_image_data_url(visual)})
                    except Exception as e:
                        eval_logger.error(f"Failed to process visual: {e}")
                    visual_groups.append(visual_group)
                    processed_visuals.extend(visual_group)

                if not self.interleave_visuals:
                    context = re.sub(r"<image \d+>", "", context)
                    content = processed_visuals + [{"type": "text", "text": context}]
                else:
                    image_placeholders = re.findall(r"<image (\d+)>", context)
                    if not image_placeholders:
                        content = processed_visuals + [{"type": "text", "text": context}]
                    else:
                        content = []
                        text_parts = re.split(r"<image \d+>", context)
                        if text_parts[0]:
                            content.append({"type": "text", "text": text_parts[0]})
                        for placeholder_idx, image_number in enumerate(image_placeholders):
                            visual_idx = int(image_number) - 1
                            if 0 <= visual_idx < len(visual_groups):
                                content.extend(visual_groups[visual_idx])
                            else:
                                eval_logger.warning(f"Image index {image_number} out of range for {len(visual_groups)} visual(s)")
                            if text_parts[placeholder_idx + 1]:
                                content.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append({"role": "user", "content": content})
                batched_messages.append(message)

            inputs = self.processor.apply_chat_template(
                batched_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "text_kwargs": {"padding": True, "pad_to_multiple_of": 8},
                    "images_kwargs": {"max_soft_tokens": self.max_soft_tokens},
                    "videos_kwargs": {
                        "max_soft_tokens": self.max_soft_tokens,
                        "num_frames": self.max_num_frames,
                        "do_sample_frames": True,
                    },
                },
            ).to(self.model.device, dtype=torch.bfloat16)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "top_k": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            cont = self.model.generate(
                **inputs,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                top_k=current_gen_kwargs["top_k"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

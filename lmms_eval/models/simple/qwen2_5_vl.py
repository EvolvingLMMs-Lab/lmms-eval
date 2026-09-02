import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


_VISUAL_PLACEHOLDER_RE = re.compile(r"<(image|video)(?:\s+(\d+))?>", re.IGNORECASE)


def _strip_visual_placeholders(context: str) -> str:
    """Remove task-level media markers after media has been attached."""
    return _VISUAL_PLACEHOLDER_RE.sub("", context)


def _interleave_visual_content(context: str, processed_visuals: list) -> list:
    """Insert visuals at ``<image>``/``<video>`` markers in prompt order.

    Unnumbered markers consume visuals sequentially. Numbered markers such as
    ``<image 2>`` retain the existing explicit-index behavior. Tasks without
    markers keep the standard visuals-first layout.
    """
    matches = list(_VISUAL_PLACEHOLDER_RE.finditer(context))
    if not matches:
        return [*processed_visuals, {"type": "text", "text": context}]

    content_parts = []
    cursor = 0
    next_visual_index = 0
    used_visual_indices = set()

    for match in matches:
        if match.start() > cursor:
            content_parts.append({"type": "text", "text": context[cursor : match.start()]})

        explicit_index = match.group(2)
        if explicit_index is not None:
            visual_index = int(explicit_index) - 1
        else:
            while next_visual_index in used_visual_indices:
                next_visual_index += 1
            visual_index = next_visual_index
            next_visual_index += 1

        if 0 <= visual_index < len(processed_visuals):
            content_parts.append(processed_visuals[visual_index])
            used_visual_indices.add(visual_index)

        cursor = match.end()

    if cursor < len(context):
        content_parts.append({"type": "text", "text": context[cursor:]})

    return content_parts or [{"type": "text", "text": ""}]


def _limit_video_inputs(video_inputs, max_num_frames: int):
    """Uniformly cap every video in a batch while preserving both endpoints."""
    if video_inputs is None or max_num_frames is None or max_num_frames <= 0:
        return video_inputs

    for video_index, video_input in enumerate(video_inputs):
        total_frames = video_input.shape[0]
        if total_frames <= max_num_frames:
            continue
        indices = np.linspace(0, total_frames - 1, max_num_frames, dtype=int)
        video_inputs[video_index] = video_input[np.unique(indices)]
    return video_inputs


def _build_video_content(video: str, *, min_pixels: int, max_pixels: int, fps: Optional[float], nframes: Optional[int]) -> dict:
    """Build qwen-vl-utils video input with an explicit sampling policy."""
    if fps is not None and nframes is not None:
        raise ValueError("Qwen video inputs cannot specify both fps and nframes")

    content = {
        "type": "video",
        "video": video,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
    }
    if fps is not None:
        content["fps"] = fps
    if nframes is not None:
        content["nframes"] = nframes
    return content


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        video_nframes: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
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

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.image_min_pixels = image_min_pixels if image_min_pixels is not None else min_pixels
        self.image_max_pixels = image_max_pixels if image_max_pixels is not None else max_pixels
        self.video_min_pixels = video_min_pixels if video_min_pixels is not None else min_pixels
        self.video_max_pixels = video_max_pixels if video_max_pixels is not None else max_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.video_nframes = video_nframes

        if self.fps is not None and self.video_nframes is not None:
            raise ValueError("Qwen video inputs cannot specify both fps and video_nframes")

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=self.image_max_pixels, min_pixels=self.image_min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

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
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _encode_image_data_url(self, image: Image.Image) -> str:
        return encode_image_to_data_url(
            image,
            image_format="JPEG",
            mime_type="image/jpeg",
            convert_rgb=True,
            quality=85,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            batched_messages = []
            for i, context in enumerate(contexts):
                message = []
                if self.system_prompt:
                    message.append({"role": "system", "content": self.system_prompt})
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            processed_visuals.append(
                                _build_video_content(
                                    visual,
                                    min_pixels=self.video_min_pixels,
                                    max_pixels=self.video_max_pixels,
                                    fps=self.fps,
                                    nframes=self.video_nframes,
                                )
                            )
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": self._encode_image_data_url(visual),
                                    "max_pixels": self.image_max_pixels,
                                    "min_pixels": self.image_min_pixels,
                                }
                            )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": _strip_visual_placeholders(context)}],
                        }
                    )
                else:
                    message.append(
                        {
                            "role": "user",
                            "content": _interleave_visual_content(context, processed_visuals),
                        }
                    )

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(batched_messages)
            video_inputs = _limit_video_inputs(video_inputs, self.max_num_frames)
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 32768,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Response: {ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        _metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_to_text,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                contexts = []
                visuals_list = []

                if round_idx != 0:
                    (
                        visuals_list,
                        contexts,
                        batched_terminal_signal,
                        batched_round_res,
                        batched_previous_round_info,
                    ) = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    batched_round_res = list(zip(*batched_round_res))
                    if batched_terminal_signal[0]:
                        break
                else:
                    visuals_list = batched_visuals
                    contexts = list(batched_contexts)

                batched_messages = []
                for i, context in enumerate(contexts):
                    message = []
                    if self.system_prompt:
                        message.append({"role": "system", "content": self.system_prompt})
                    if self.reasoning_prompt:
                        context = context.strip() + self.reasoning_prompt

                    processed_visuals = []
                    if visuals_list[i] is not None:
                        for visual in visuals_list[i]:
                            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                                processed_visuals.append(
                                    _build_video_content(
                                        visual,
                                        min_pixels=self.video_min_pixels,
                                        max_pixels=self.video_max_pixels,
                                        fps=self.fps,
                                        nframes=self.video_nframes,
                                    )
                                )
                            elif isinstance(visual, Image.Image):
                                processed_visuals.append(
                                    {
                                        "type": "image",
                                        "image": self._encode_image_data_url(visual),
                                        "max_pixels": self.image_max_pixels,
                                        "min_pixels": self.image_min_pixels,
                                    }
                                )

                    if self.interleave_visuals is False:
                        message.append(
                            {
                                "role": "user",
                                "content": processed_visuals + [{"type": "text", "text": _strip_visual_placeholders(context)}],
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": _interleave_visual_content(context, processed_visuals),
                            }
                        )

                    batched_messages.append(message)

                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
                image_inputs, video_inputs = process_vision_info(batched_messages)
                video_inputs = _limit_video_inputs(video_inputs, self.max_num_frames)
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                default_gen_kwargs = {
                    "max_new_tokens": 32768,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
                pad_token_id = self.tokenizer.pad_token_id

                if current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                clean_answers = []
                for ans in answers:
                    clean_answers.append(ans)

                batched_round_res.append(clean_answers)
                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial(
                "generate_until_multi_round",
                (batched_contexts[0], gen_kwargs),
                batched_round_res,
            )
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

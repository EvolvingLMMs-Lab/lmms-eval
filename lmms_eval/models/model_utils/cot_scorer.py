import copy
import json
import os
import re
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from decord import VideoReader, cpu
from packaging import version
from PIL import Image
from transformers import AutoConfig



REASONER_PROMPT_TEMPLATE = """You are a video understanding expert. Your task is to analyze whether a given
video frame may help answer a specific question about a video.
You are provided with a single frame from the video and should determine its
potential relevance to the question based only on the visual content of
this frame.
Question:
"{query}"
Frame Information:
- Total video duration: {total_time}
- Frame index: {frame_idx}
- Frame timestamp: {frame_time}
Task:
Analyze whether this frame might be helpful for answering the question, even
partially. Focus only on what can be observed in this frame.
Please provide your analysis in the following format:
#FRAME {frame_idx}:
This frame shows [...]. It might help understand [...].
I (slightly/highly) recommend selecting this frame / I (slightly/highly) do
not recommend selecting this frame because [...]. / I am uncertain because
[...].
Notes:
- The frame does not need to contain the final answer.
- Frames that provide partial evidence, context, or clues may still be useful.
"""

SCORER_PROMPT_IMAGE_ONLY = """You are a component of a video question answering system. Your task is to
evaluate how helpful a specific video frame is for answering a given
question.
You are provided with:
1. The question
2. A video frame (image)
Question:
{query}
Task:
Based on the visual content of the frame, assign a helpfulness score between
0.000 and 1.000, indicating how useful this frame may be for answering the
question.
Consider the following aspects:
- Whether the visual content is relevant to the question
- Whether the frame provides partial or direct evidence

Output Format:
Output a single real number between 0.00 and 1.00 (e.g., 0.87).
Do not include any explanation or additional text.
"""

SCORER_PROMPT_TEXT_ONLY = """You are a component of a video question answering system. Your task is to
evaluate how helpful a specific video frame is for answering a given
question.
You are provided with:
1. The question
2. A textual analysis describing the content and potential relevance of the
frame
Question:
{query}
Frame Analysis:
{rationale}
Task:
Based on the provided analysis, assign a helpfulness score between 0.000 and
1.000, indicating how useful this frame may be for answering the question.
Consider the following aspects:
- Whether the analysis is relevant to the question
- Whether it provides partial or direct evidence
- Whether the analysis supports a logical reasoning process
Output Format:
Output a single real number between 0.000 and 1.000 (e.g., 0.875).
Do not include any explanation or additional text.
"""

SCORER_PROMPT_IMAGE_AND_TEXT = """You are a component of a video question answering system. Your task is to
evaluate how helpful a specific video frame is for answering a given
question.
You are provided with:
1. The question
2. A video frame (image)
3. A textual analysis describing the content and potential relevance of the
frame
Question:
{query}
Frame Analysis:
{rationale}
Task:
Based on both the visual content of the frame and the provided analysis,
assign a helpfulness score between 0 and 1, indicating how useful
this frame may be for answering the question.
Consider the following aspects:
- Whether the visual content is relevant to the question
- Whether the frame provides partial or direct evidence
- Whether the frame is clear and unambiguous
- Whether the analysis supports a logical reasoning process
Output Format:
Output a single real number between 0.00 and 1.00 (e.g., 0.88).
Do not include any explanation or additional text.
"""

# Determine best attention implementation for LLaVA-OneVision
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


def build_reasoner_prompt(
    *,
    query: str,
    total_time: str,
    frame_idx: int,
    frame_time: str,
) -> str:
    return REASONER_PROMPT_TEMPLATE.format(
        query=query,
        total_time=total_time,
        frame_idx=frame_idx,
        frame_time=frame_time,
    )


def build_scorer_prompt(
    *,
    query: str,
    rationale: str,
    scorer_type: int,
) -> str:
    if scorer_type == 0:
        return SCORER_PROMPT_IMAGE_ONLY.format(query=query)
    if scorer_type == 1:
        return SCORER_PROMPT_TEXT_ONLY.format(query=query, rationale=rationale)
    return SCORER_PROMPT_IMAGE_AND_TEXT.format(query=query, rationale=rationale)


def _sanitize_reasoner_name(name: str) -> str:
    return name.replace("/", "")


def _sanitize_scorer_name(name: str) -> str:
    return name.replace("/", "").replace("_", "")


class FrameCoTScorer:
    def __init__(
        self,
        *,
        scorer_type: int,
        reasoner_name: str,
        scorer_name: str,
        candidates: int,
        task_name: str,
        cache_root: str = "./score_cache",
        use_cache: bool = True,
        video_loader: Optional[Callable[[str, int], Any]] = None,
        device: str = "cuda:0",
        reasoner_max_new_tokens: int = 512,
        scorer_max_new_tokens: int = 16,
        temperature: float = 0.1,
        llava_conv_template: str = "qwen_1_5",
        llava_attn_implementation: str = "eager",
        debug_print: bool = False,
    ) -> None:
        if scorer_type not in (0, 1, 2):
            raise ValueError(f"Invalid scorer_type: {scorer_type}")
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        if not scorer_name:
            raise ValueError("scorer_name is required")
        if scorer_type in (1, 2) and not reasoner_name:
            raise ValueError("reasoner_name is required when scorer_type needs rationale")

        self.scorer_type = scorer_type
        self.reasoner_name = reasoner_name
        self.scorer_name = scorer_name
        self.candidates = candidates
        self.task_name = task_name
        self.cache_root = cache_root
        self.use_cache = use_cache
        self.video_loader = video_loader or self._load_video_frames
        self.device = device
        self.reasoner_max_new_tokens = reasoner_max_new_tokens
        self.scorer_max_new_tokens = scorer_max_new_tokens
        self.temperature = temperature
        self.llava_conv_template = llava_conv_template
        self.llava_attn_implementation = llava_attn_implementation
        self.debug_print = debug_print
        self._backend_cache: Dict[str, Any] = {}

        os.makedirs(self._cache_dir(), exist_ok=True)

    def _cache_dir(self) -> str:
        reasoner = _sanitize_reasoner_name(self.reasoner_name)
        scorer = _sanitize_scorer_name(self.scorer_name)
        subdir = f"{self.task_name}_{reasoner}_{scorer}_{self.candidates}"
        return os.path.join(self.cache_root, subdir)

    def _cache_path(self, video_path: str) -> str:
        base = os.path.basename(video_path)
        return os.path.join(self._cache_dir(), f"{base}.json")

    def _load_cache(self, video_path: str) -> Dict[str, Any]:
        cache_path = self._cache_path(video_path)
        if not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self, video_path: str, key: str, value: Dict[str, Any]) -> None:
        cache_path = self._cache_path(video_path)
        existing = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing[key] = value
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def _load_video_frames(self, video_path: str, num_frames: int) -> Tuple[np.ndarray, list, str]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()
        duration_seconds = total_frame_num / fps if fps else 0.0
        frame_idx = np.linspace(0, total_frame_num - 1, num_frames, dtype=int).tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        frame_times = [self._format_time(i / fps) if fps else "unknown" for i in frame_idx]
        return frames, frame_times, self._format_time(duration_seconds)

    def _format_time(self, seconds: float) -> str:
        if seconds <= 0:
            return "0:00"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        hours = minutes // 60
        if hours > 0:
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def generate_reasoning(
        self,
        frame: np.ndarray,
        query: str,
        *,
        frame_idx: int,
        frame_time: str,
        total_time: str,
    ) -> str:
        prompt = build_reasoner_prompt(
            query=query,
            total_time=total_time,
            frame_idx=frame_idx,
            frame_time=frame_time,
        )
        output = self._auto_reasoner(frame=frame, prompt=prompt)
        return "" if output is None else str(output)

    def score_frame(
        self,
        frame: np.ndarray,
        query: str,
        reasoning_rationale: str,
        *,
        frame_idx: int,
    ) -> float:
        prompt = build_scorer_prompt(query=query, rationale=reasoning_rationale, scorer_type=self.scorer_type)
        
        return self._auto_score(frame=frame, rationale=reasoning_rationale, prompt=prompt)

    def score_video(
        self,
        *,
        video_path: str,
        query: str,
        query_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = query_id or query
        if self.use_cache:
            cached = self._load_cache(video_path)
            if key in cached:
                return {
                    "meta": self.meta(),
                    "frames": cached[key],
                }

        loaded = self.video_loader(video_path, self.candidates)
        if isinstance(loaded, tuple) and len(loaded) == 3:
            frames, frame_times, total_time = loaded
        else:
            frames = loaded
            frame_times = ["unknown"] * len(frames)
            total_time = "unknown"
        frames_map: Dict[str, Dict[str, Any]] = {}

        for idx, frame in enumerate(frames):
            if self.scorer_type == 0:
                reasoning_rationale = ""
            else:
                reasoning_rationale = self.generate_reasoning(
                    frame,
                    query,
                    frame_idx=idx,
                    frame_time=frame_times[idx],
                    total_time=total_time,
                )
            score = self.score_frame(frame, query, reasoning_rationale, frame_idx=idx)
            frames_map[str(idx)] = {
                "frame_time": frame_times[idx],
                "reasoning_rationale": reasoning_rationale,
                "score": score,
            }

        if self.use_cache:
            self._save_cache(video_path, key, frames_map)

        return {
            "meta": self.meta(),
            "frames": frames_map,
        }

    def meta(self) -> Dict[str, Any]:
        return {
            "scorer_type": self.scorer_type,
            "reasoner": self.reasoner_name,
            "scorer": self.scorer_name,
            "candidates": self.candidates,
            "task_name": self.task_name,
        }

    def _auto_reasoner(self, *, frame: np.ndarray, prompt: str) -> str:
        backend = self._get_backend(self.reasoner_name)
        return backend.generate_text(
            prompt=prompt,
            frame=frame,
            max_new_tokens=self.reasoner_max_new_tokens,
            temperature=self.temperature,
        )

    def _auto_score(self, *, frame: np.ndarray, rationale: str, prompt: str) -> float:
        backend = self._get_backend(self.scorer_name)
        text = backend.generate_text(
            prompt=prompt,
            frame=None if self.scorer_type == 1 else frame,
            max_new_tokens=self.scorer_max_new_tokens,
            temperature=self.temperature,
        )
        if self.debug_print:
            print(f"[cot_scorer] raw_score_output: {text}")
        return _parse_score(text)

    def _get_backend(self, name: str) -> Any:
        if name in self._backend_cache:
            return self._backend_cache[name]
        backend_type = _infer_backend(name)
        if backend_type == "llava_onevision":
            backend = _LlavaOneVisionBackend(
                model_name=name,
                device=self.device,
                conv_template=self.llava_conv_template,
                attn_implementation=self.llava_attn_implementation,
            )
        elif backend_type == "llava":
            backend = _LlavaBackend(
                model_name=name,
                device=self.device,
                conv_template=self.llava_conv_template,
                attn_implementation=self.llava_attn_implementation,
            )
        else:
            backend = _QwenBackend(model_name=name, device=self.device)
        self._backend_cache[name] = backend
        return backend

def _infer_backend(name: str) -> str:
    lowered = name.lower()
    if "llava_onevision" in lowered or "llava-onevision" in lowered:
        if "onevision1_5" not in lowered and "onevision-1.5" not in lowered:
            return "llava_onevision"
    if "llava" in lowered:
        return "llava"
    return "qwen"


def _parse_score(text: str) -> float:
    match = re.findall(r"\d+\.?\d*", text)
    if not match:
        return 0.0
    score = float(match[0])
    return min(max(score, 0.0), 1.0)


class _QwenBackend:
    def __init__(self, *, model_name: str, device: str) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2VLForConditionalGeneration

        self.model_name = model_name
        self.device = device
        self.is_qwen2_vl = "Qwen2-VL" in model_name.lower() or "qwen2_vl" in model_name.lower() or "qwen2vl" in model_name.lower()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.is_qwen2_vl:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
                low_cpu_mem_usage=True,
            )
        self.model.eval()

    def generate_text(
        self,
        *,
        prompt: str,
        frame: Optional[np.ndarray],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        if frame is not None:
            pil_image = Image.fromarray(frame.astype("uint8"))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


class _LlavaBackend:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        conv_template: str,
        attn_implementation: str,
    ) -> None:
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )

        self.model_name = model_name
        self.device = device
        self.conv_template = conv_template
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._conv_templates = conv_templates

        model_name_resolved = get_model_name_from_path(self.model_name)
        self.tokenizer, self.model, self.image_processor, self.model_config = load_pretrained_model(
            self.model_name,
            None,
            model_name_resolved,
            device_map=self.device,
            attn_implementation=attn_implementation,
        )
        if self.device != "auto":
            try:
                vision_tower = self.model.get_vision_tower()
                vision_tower.to(device=self.device, dtype=torch.float16)
            except Exception:
                pass
        self.model.eval()

    def generate_text(
        self,
        *,
        prompt: str,
        frame: Optional[np.ndarray],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        image_tensor = None
        image_sizes = None
        if frame is not None:
            pil_image = Image.fromarray(frame.astype("uint8"))
            visual = [pil_image]
            if getattr(self.model_config, "image_aspect_ratio", None) is None:
                self.model_config.image_aspect_ratio = "pad"
            image_tensor = self._process_images(visual, self.image_processor, self.model_config)
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            image_sizes = [[pil_image.size[0], pil_image.size[1]]]
            if getattr(self.model_config, "mm_use_im_start_end", False):
                prompt_input = f"{self._DEFAULT_IM_START_TOKEN}{self._DEFAULT_IMAGE_TOKEN}{self._DEFAULT_IM_END_TOKEN}\n{prompt}"
            else:
                prompt_input = f"{self._DEFAULT_IMAGE_TOKEN}\n{prompt}"
        else:
            prompt_input = prompt

        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(self._conv_templates[self.conv_template])
        else:
            conv = self._conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], prompt_input)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = outputs[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


class _LlavaOneVisionBackend:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        conv_template: str,
        attn_implementation: Optional[str],
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_mode: str = "bilinear",
    ) -> None:
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IGNORE_INDEX,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model

        self.model_name = model_name
        self.conv_template = conv_template

        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._conv_templates = conv_templates

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self.device = torch.device(device)
            self.device_map = device

        resolved_attn = attn_implementation or best_fit_attn_implementation

        llava_model_args = {"multimodal": True, "attn_implementation": resolved_attn}
        overwrite_config = {
            "mm_spatial_pool_stride": mm_spatial_pool_stride,
            "mm_spatial_pool_mode": mm_spatial_pool_mode,
        }
        AutoConfig.from_pretrained(self.model_name)
        llava_model_args["overwrite_config"] = overwrite_config

        model_name_resolved = get_model_name_from_path(self.model_name)
        try:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_name,
                None,
                model_name_resolved,
                device_map=self.device_map,
                **llava_model_args,
            )
        except TypeError:
            llava_model_args.pop("multimodal", None)
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_name,
                None,
                model_name_resolved,
                device_map=self.device_map,
                **llava_model_args,
            )
        self.model_config = self.model.config
        if self.device_map != "auto":
            try:
                vision_tower = self.model.get_vision_tower()
                vision_tower.to(device=self.device, dtype=torch.float16)
            except Exception:
                pass
        self.model.eval()

    def generate_text(
        self,
        *,
        prompt: str,
        frame: Optional[np.ndarray],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        image_tensor = None
        image_sizes = None
        if frame is not None:
            pil_image = Image.fromarray(frame.astype("uint8"))
            visual = [pil_image]
            if getattr(self.model_config, "image_aspect_ratio", None) is None:
                self.model_config.image_aspect_ratio = "pad"
            image_tensor = self._process_images(visual, self.image_processor, self.model_config)
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            image_sizes = [[pil_image.size[0], pil_image.size[1]]]
            if getattr(self.model_config, "mm_use_im_start_end", False):
                prompt_input = f"{self._DEFAULT_IM_START_TOKEN}{self._DEFAULT_IMAGE_TOKEN}{self._DEFAULT_IM_END_TOKEN}\n{prompt}"
            else:
                prompt_input = f"{self._DEFAULT_IMAGE_TOKEN}\n{prompt}"
        else:
            prompt_input = prompt

        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(self._conv_templates[self.conv_template])
        else:
            conv = self._conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], prompt_input)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
               input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = outputs[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

"""SGLang Diffusion backend for image/video generation benchmarks.

This adapter uses SGLang's in-process ``DiffGenerator`` and emits generated
media in lmms-eval's JSON output format.  It is primarily intended for Wan
image-to-video models, including ``Wan2.2-I2V-A14B-Diffusers``.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.progress import make_progress
from lmms_eval.protocol import ChatMessages

_DEFAULT_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
_DEFAULT_PROMPT = "Generate a natural video continuation of this image."
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

_LANGUAGE_RUNTIMES = {"language", "llm", "vlm", "srt"}
_DIFFUSION_RUNTIMES = {"diffusion", "diffusion_runtime", "multimodal_gen", "video", "wan"}

_SAMPLING_KEYS = {
    "adjust_frames",
    "boundary_ratio",
    "cfg_normalization",
    "enable_frame_interpolation",
    "enable_sequence_shard",
    "enable_teacache",
    "enable_upscaling",
    "fps",
    "frame_interpolation_exp",
    "frame_interpolation_model_path",
    "frame_interpolation_scale",
    "guidance_rescale",
    "guidance_scale",
    "guidance_scale_2",
    "height",
    "negative_prompt",
    "num_frames",
    "num_frames_round_down",
    "num_inference_steps",
    "num_outputs_per_prompt",
    "output_compression",
    "output_quality",
    "seed",
    "suppress_logs",
    "teacache_params",
    "true_cfg_scale",
    "upscaling_model_path",
    "upscaling_scale",
    "width",
}


def _safe(value: Any, default: str = "x") -> str:
    normalized = _SAFE_RE.sub("_", str(value)).strip("_") or default
    return normalized[:128]


def _default_output_dir(model_path: str) -> str:
    model_slug = _safe(os.path.basename(str(model_path).rstrip("/")), default="model")
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    return os.path.join("./logs/sglang_diffusion", model_slug, run_id)


def _normalize_runtime(runtime: Any) -> str:
    normalized = str(runtime or "auto").strip().lower().replace("-", "_")
    if normalized == "auto" or normalized in _LANGUAGE_RUNTIMES or normalized in _DIFFUSION_RUNTIMES:
        return normalized
    choices = "auto, language, diffusion"
    raise ValueError(f"Unknown SGLang runtime {runtime!r}; expected one of: {choices}")


def _local_model_index(model_path: str) -> dict[str, Any] | None:
    path = Path(os.path.expandvars(os.path.expanduser(str(model_path)))) / "model_index.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _infer_output_type(model_path: str) -> str:
    model_index = _local_model_index(model_path) or {}
    descriptor = " ".join((str(model_path), str(model_index.get("_class_name") or ""))).lower()
    video_markers = ("video", "wan", "i2v", "t2v", "ti2v", "ltx")
    return "video" if any(marker in descriptor for marker in video_markers) else "image"


def is_sglang_diffusion_model(model_path: str, runtime: Any = "auto") -> bool:
    """Return whether ``--model sglang`` should use SGLang Diffusion.

    ``runtime=diffusion`` and ``runtime=language`` are explicit overrides. In
    auto mode, local Diffusers metadata and well-known diffusion repository
    naming conventions are used without making a network request.
    """

    normalized_runtime = _normalize_runtime(runtime)
    if normalized_runtime in _DIFFUSION_RUNTIMES:
        return True
    if normalized_runtime in _LANGUAGE_RUNTIMES:
        return False

    model_index = _local_model_index(model_path)
    if model_index is not None:
        class_name = str(model_index.get("_class_name") or "").lower()
        if "pipeline" in class_name and any(token in class_name for token in ("diffusion", "image", "video", "wan", "flux")):
            return True
        components = set(model_index)
        if "scheduler" in components and ({"transformer", "unet", "vae"} & components):
            return True

    normalized_model = str(model_path).strip().lower().rstrip("/")
    model_name = normalized_model.rsplit("/", 1)[-1]
    return normalized_model.startswith("wan-ai/wan") or model_name.startswith("wan") or model_name.endswith("-diffusers") or model_name.startswith(("flux.", "qwen-image"))


def _load_diff_generator():
    try:
        from sglang.multimodal_gen import DiffGenerator
    except (ImportError, AttributeError) as exc:
        raise ImportError("SGLang Diffusion is required for Wan generation. Install the validated build with `uv pip install --prerelease=allow 'sglang[diffusion]==0.5.10.post1'`.") from exc
    return DiffGenerator


def _as_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported conditioning image type: {type(image).__name__}")

    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    image = np.squeeze(image)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Conditioning image must have 2 or 3 dimensions, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.asarray(image)
        if image.size and float(np.nanmax(image)) <= 1.0 and float(np.nanmin(image)) >= 0.0:
            image = image * 255
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


@dataclass
class _PreparedDiffusionRequest:
    sampling_params: dict[str, Any]
    expected_output_path: str


@register_model("sglang_diffusion", "sglang-diffusion", "sglang_wan", "sglang-wan")
class SGLangDiffusion(lmms):
    """lmms-eval adapter around ``sglang.multimodal_gen.DiffGenerator``."""

    is_simple = False

    def __init__(
        self,
        model: Optional[str] = None,
        pretrained: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        num_gpus: Optional[int] = None,
        batch_size: int = 1,
        output_dir: Optional[str] = None,
        output_path: Optional[str] = None,
        num_frames: Optional[int] = None,
        nframes: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_2: Optional[float] = None,
        fps: Optional[int] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        local_mode: bool = True,
        backend: str = "auto",
        runtime: Optional[str] = None,
        output_type: str = "auto",
        trust_remote_code: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if runtime is not None and not is_sglang_diffusion_model(model or pretrained or _DEFAULT_MODEL, runtime=runtime):
            raise ValueError("SGLangDiffusion cannot be used with runtime=language")

        if model and pretrained and model != pretrained:
            raise ValueError(f"Conflicting SGLang model paths: model={model!r}, pretrained={pretrained!r}")
        self.model_path = str(model or pretrained or _DEFAULT_MODEL)
        self._model = self.model_path
        self._config = None

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            raise RuntimeError("SGLang Diffusion manages its own GPU workers. Launch lmms-eval as one process and set num_gpus in --model_args instead of using accelerate launch.")
        self.accelerator = accelerator
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        self.device = device or accelerator.device
        self.batch_size_per_gpu = int(batch_size)

        selected_num_frames = num_frames if num_frames is not None else nframes
        self.num_frames = None if selected_num_frames is None else int(selected_num_frames)
        self.height = None if height is None else int(height)
        self.width = None if width is None else int(width)
        self.num_inference_steps = None if num_inference_steps is None else int(num_inference_steps)
        self.guidance_scale = None if guidance_scale is None else float(guidance_scale)
        self.guidance_scale_2 = None if guidance_scale_2 is None else float(guidance_scale_2)
        self.fps = None if fps is None else int(fps)
        self.seed = None if seed is None else int(seed)
        self.negative_prompt = negative_prompt
        model_index = _local_model_index(self.model_path) or {}
        model_descriptor = " ".join((self.model_path, str(model_index.get("_class_name") or ""))).lower()
        self._requires_image = "i2v" in model_descriptor or "image-to-video" in model_descriptor or "imagetovideo" in model_descriptor
        normalized_output_type = str(output_type or "auto").strip().lower()
        if normalized_output_type == "auto":
            normalized_output_type = _infer_output_type(self.model_path)
        if normalized_output_type not in {"image", "video"}:
            raise ValueError(f"output_type must be auto, image, or video; got {output_type!r}")
        self.output_type = normalized_output_type
        self.output_key = f"{normalized_output_type}s"
        self.output_file_name = f"output.{('mp4' if normalized_output_type == 'video' else 'png')}"

        selected_output_dir = output_dir or output_path or _default_output_dir(self.model_path)
        self.output_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(str(selected_output_dir))))
        os.makedirs(self.output_dir, exist_ok=True)

        selected_num_gpus = num_gpus if num_gpus is not None else tensor_parallel_size
        self.num_gpus = int(selected_num_gpus or 1)
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be at least 1, got {self.num_gpus}")

        # Translate common lmms/FastVideo spellings to SGLang ServerArgs.
        if "data_parallel" in kwargs and "dp_size" not in kwargs:
            kwargs["dp_size"] = kwargs.pop("data_parallel")
        if "sp_size" in kwargs and "sp_degree" not in kwargs:
            kwargs["sp_degree"] = kwargs.pop("sp_size")
        ignored = {
            "chat_template",
            "gpu_memory_utilization",
            "json_model_override_args",
            "max_frame_num",
            "max_pixels",
            "min_pixels",
            "threads",
            "work_dir",
        }
        ignored_present = sorted(key for key in ignored if kwargs.pop(key, None) is not None)
        if ignored_present:
            eval_logger.warning(f"Ignoring language-runtime-only SGLang options for diffusion: {', '.join(ignored_present)}")
        if kwargs.pop("mcp_server_path", None) is not None:
            raise ValueError("MCP tool calling is not supported by the SGLang Diffusion backend")
        kwargs.pop("max_turn", None)

        server_kwargs = {
            "model_path": self.model_path,
            "num_gpus": self.num_gpus,
            "output_path": self.output_dir,
            "backend": backend,
            "trust_remote_code": bool(trust_remote_code),
            **kwargs,
        }
        generator_cls = _load_diff_generator()
        self.client = generator_cls.from_pretrained(local_mode=bool(local_mode), **server_kwargs)
        self._closed = False
        atexit.register(self.close)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return None

    @property
    def model(self):
        return self.client

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not defined for diffusion generation")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is not supported by SGLang Diffusion")

    @staticmethod
    def _extract_prompt(chat_messages: ChatMessages) -> str:
        texts: list[str] = []
        for message in chat_messages.messages:
            if message.role != "user":
                continue
            texts.extend(content.text for content in message.content if content.type == "text" and content.text)
        return "\n".join(texts).strip()

    def _request_output_dir(self, task: Any, split: Any, doc_id: Any) -> str:
        request_dir = os.path.join(self.output_dir, _safe(task, "task"), _safe(split, "split"), _safe(doc_id, "doc"))
        os.makedirs(request_dir, exist_ok=True)
        return os.path.abspath(request_dir)

    def _materialize_image(self, image: Any, request_dir: str) -> str:
        if isinstance(image, dict):
            for key in ("path", "image", "url"):
                if key in image:
                    image = image[key]
                    break
        if isinstance(image, str):
            if image.startswith("file://"):
                image = image[7:]
            if image.startswith(("http://", "https://", "data:")):
                return image
            expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(image)))
            if not os.path.isfile(expanded):
                raise FileNotFoundError(f"Conditioning image does not exist: {expanded}")
            return expanded

        image_path = os.path.join(request_dir, "input.png")
        _as_pil_image(image).save(image_path, format="PNG")
        return os.path.abspath(image_path)

    def make_one_request(self, request: Instance) -> _PreparedDiffusionRequest:
        _ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        if task is None:
            if len(self.task_dict) != 1:
                raise KeyError(f"Request omitted task name while {len(self.task_dict)} tasks are loaded")
            task = next(iter(self.task_dict))
        if split is None:
            task_splits = self.task_dict[task]
            if len(task_splits) != 1:
                raise KeyError(f"Request omitted split while task {task!r} has multiple splits")
            split = next(iter(task_splits))

        doc = self.task_dict[task][split][doc_id]
        chat_messages = ChatMessages(messages=doc_to_messages(doc))
        images, _videos, _audios = chat_messages.extract_media()
        prompt = self._extract_prompt(chat_messages) or _DEFAULT_PROMPT
        request_dir = self._request_output_dir(task, split, doc_id)

        sampling_params: dict[str, Any] = {
            "prompt": prompt,
            "output_path": request_dir,
            "output_file_name": self.output_file_name,
            "save_output": True,
            "return_file_paths_only": True,
        }
        model_sampling_defaults = {
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "guidance_scale_2": self.guidance_scale_2,
            "fps": self.fps,
            "seed": self.seed,
        }
        sampling_params.update({key: value for key, value in model_sampling_defaults.items() if value is not None})
        if self.negative_prompt is not None:
            sampling_params["negative_prompt"] = self.negative_prompt
        if images:
            sampling_params["image_path"] = self._materialize_image(images[0], request_dir)
        elif self._requires_image:
            raise ValueError(f"{self.model_path} requires a conditioning image, but the request contained none")

        for key, value in dict(gen_kwargs or {}).items():
            if key in _SAMPLING_KEYS and value is not None:
                sampling_params[key] = value

        expected_output_path = os.path.join(request_dir, self.output_file_name)
        return _PreparedDiffusionRequest(sampling_params=sampling_params, expected_output_path=expected_output_path)

    @staticmethod
    def _result_paths(result: Any) -> list[str]:
        results = result if isinstance(result, list) else ([result] if result is not None else [])
        paths: list[str] = []
        for item in results:
            path = getattr(item, "output_file_path", None)
            if path:
                resolved = os.path.abspath(os.path.expanduser(str(path)))
                if os.path.isfile(resolved):
                    paths.append(resolved)
        return paths

    @staticmethod
    def _pack_result(paths: list[str], output_key: str = "videos") -> GenerationResult:
        return GenerationResult(
            text=json.dumps({"text": "", output_key: paths}, ensure_ascii=False, separators=(",", ":")),
            token_counts=TokenCounts(),
        )

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        responses: list[GenerationResult] = []
        total_elapsed = 0.0
        generated_outputs = 0
        progress = make_progress(total=len(requests), disable=(self.rank != 0), desc="SGLang Diffusion generating")

        for request in requests:
            started_at = time.perf_counter()
            try:
                prepared = self.make_one_request(request)
                result = self.client.generate(sampling_params_kwargs=prepared.sampling_params)
                paths = self._result_paths(result)
                if not paths:
                    eval_logger.error(f"SGLang Diffusion did not produce an output file for {prepared.expected_output_path}")
                responses.append(self._pack_result(paths, self.output_key))
                generated_outputs += len(paths)
            except Exception as exc:  # Keep long benchmark runs alive after a bad sample.
                eval_logger.exception(f"SGLang Diffusion request failed: {exc}")
                responses.append(self._pack_result([], self.output_key))
            finally:
                total_elapsed += time.perf_counter() - started_at
                progress.update(1)

        progress.close()
        log_metrics(
            total_elapsed_time=total_elapsed,
            total_gen_tokens=0,
            avg_speed=0.0,
            additional_metrics={
                "total_requests": len(requests),
                f"generated_{self.output_key}": generated_outputs,
                f"{self.output_key}_per_second": generated_outputs / total_elapsed if total_elapsed > 0 else 0.0,
            },
        )
        return responses

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        shutdown = getattr(self.client, "shutdown", None)
        if callable(shutdown):
            shutdown()

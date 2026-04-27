from __future__ import annotations

import atexit
import copy
import importlib
import json
import os
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import is_package_available, optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

_has_transformers = is_package_available("transformers")
_has_vllm = is_package_available("vllm")
_has_vllm_omni = is_package_available("vllm_omni")
_has_soundfile = is_package_available("soundfile")
_has_diffusers = is_package_available("diffusers")

AutoProcessor = None
SamplingParams = None
Omni = None
fetch_audio = None
fetch_image = None
fetch_video = None
soundfile = None
export_to_video = None

WORKERS = int(os.getenv("WORKERS", "8"))


def _safe(name: Any, default: str = "x") -> str:
    s = "".join(ch if str(ch).isalnum() or ch in "._-" else "_" for ch in str(name)).strip("_")
    return (s or default)[:128]


def _generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{str(uuid.uuid4())[:8]}"


def _model_slug(model_path: str) -> str:
    return _safe(os.path.basename(str(model_path).rstrip("/")) or "model", default="model")


def _default_output_dir(model_path: str) -> str:
    return os.path.join("./logs/vllm_omni", _model_slug(model_path), _generate_run_id())


def _build_diffusion_parallel_config(
    tensor_parallel_size: int,
    data_parallel_size: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    parallel_config = {
        "pipeline_parallel_size": int(kwargs.pop("pipeline_parallel_size", 1) or 1),
        "data_parallel_size": int(data_parallel_size),
        "tensor_parallel_size": int(tensor_parallel_size),
        "ulysses_degree": int(kwargs.pop("ulysses_degree", 1) or 1),
        "ring_degree": int(kwargs.pop("ring_degree", 1) or 1),
        "ulysses_mode": kwargs.pop("ulysses_mode", "strict") or "strict",
        "cfg_parallel_size": int(kwargs.pop("cfg_parallel_size", 1) or 1),
        "vae_patch_parallel_size": int(kwargs.pop("vae_patch_parallel_size", 1) or 1),
        "use_hsdp": bool(kwargs.pop("use_hsdp", False)),
        "hsdp_shard_size": int(kwargs.pop("hsdp_shard_size", -1) or -1),
        "hsdp_replicate_size": int(kwargs.pop("hsdp_replicate_size", 1) or 1),
    }
    sequence_parallel_size = kwargs.pop("sequence_parallel_size", None)
    if sequence_parallel_size is not None:
        parallel_config["sequence_parallel_size"] = int(sequence_parallel_size)
    return parallel_config


def _read_model_index_float(model_path: str, key: str) -> float | None:
    model_index_path = os.path.join(os.path.expanduser(str(model_path)), "model_index.json")
    if not os.path.isfile(model_index_path):
        return None
    try:
        with open(model_index_path, "r", encoding="utf-8") as handle:
            value = json.load(handle).get(key)
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _PreparedRequest:
    prompt: dict[str, Any]
    sampling_params_list: Sequence[Any]
    task: str
    split: Any
    doc_id: Any


@register_model("vllm_omni", "vllm-omni")
class VLLMOmni(lmms):
    is_simple = False

    @staticmethod
    def _lazy_import_runtime_dependencies() -> None:
        global AutoProcessor, SamplingParams, Omni, fetch_audio, fetch_image, fetch_video, soundfile, export_to_video

        if AutoProcessor is None:
            AutoProcessor, _ = optional_import("transformers", "AutoProcessor")
        if SamplingParams is None:
            SamplingParams, _ = optional_import("vllm", "SamplingParams")
        if Omni is None:
            Omni, _ = optional_import("vllm_omni", "Omni")
        if fetch_audio is None:
            fetch_audio, _ = optional_import("vllm.multimodal.utils", "fetch_audio")
        if fetch_image is None:
            fetch_image, _ = optional_import("vllm.multimodal.utils", "fetch_image")
        if fetch_video is None:
            fetch_video, _ = optional_import("vllm.multimodal.utils", "fetch_video")
        if soundfile is None and _has_soundfile:
            soundfile, _ = optional_import("soundfile")
        if export_to_video is None and _has_diffusers:
            export_to_video, _ = optional_import("diffusers.utils", "export_to_video")

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Omni-7B",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        max_frame_num: int = 32,
        trust_remote_code: bool = True,
        chat_template: Optional[str] = None,
        processor_name: Optional[str] = None,
        processor_kwargs: Optional[dict[str, Any]] = None,
        fps: Optional[int] = 16,
        nframes: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_frames: Optional[int] = None,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        boundary_ratio: Optional[float] = None,
        flow_shift: Optional[float] = None,
        output_dir: Optional[str] = None,
        output_modalities: Optional[str | list[str]] = None,
        extract_audio_from_video: bool = True,
        disable_log_stats: bool = False,
        max_new_tokens: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        self._lazy_import_runtime_dependencies()
        if not _has_vllm_omni or Omni is None:
            raise ImportError("vllm-omni is not installed. Please install `vllm-omni` first.")
        if not _has_vllm or SamplingParams is None:
            raise ImportError("vllm is required by vllm_omni.")

        self.model = model
        self.batch_size_per_gpu = int(batch_size)
        self.max_frame_num = int(max_frame_num)
        self.fps = int(fps) if fps is not None else None
        resolved_num_frames = num_frames if num_frames is not None else nframes
        self.num_frames = int(resolved_num_frames) if resolved_num_frames is not None else int(max_frame_num)
        self.nframes = self.num_frames
        self.max_new_tokens = int(max_new_tokens)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.guidance_scale_2 = None if guidance_scale_2 is None else float(guidance_scale_2)
        self.height = int(height)
        self.width = int(width)
        self.seed = int(seed)
        self.boundary_ratio = None if boundary_ratio is None else float(boundary_ratio)
        if self.boundary_ratio is None:
            self.boundary_ratio = _read_model_index_float(self.model, "boundary_ratio")
        self.flow_shift = None if flow_shift is None else float(flow_shift)
        self.extract_audio_from_video = bool(extract_audio_from_video)
        self.disable_log_stats = bool(disable_log_stats)
        self.output_modalities = self._normalize_output_modalities(output_modalities)

        self.output_dir = os.path.abspath(os.path.expanduser(output_dir or _default_output_dir(self.model)))
        os.makedirs(self.output_dir, exist_ok=True)

        processor_kwargs = self._maybe_parse_json_dict(processor_kwargs) or {}
        kwargs = self._maybe_parse_json_like_kwargs(kwargs)
        if "parallel_config" not in kwargs or kwargs["parallel_config"] is None:
            kwargs["parallel_config"] = _build_diffusion_parallel_config(
                tensor_parallel_size=tensor_parallel_size,
                data_parallel_size=data_parallel_size,
                kwargs=kwargs,
            )
        if "log_stats" not in kwargs:
            kwargs["log_stats"] = not self.disable_log_stats

        self.processor = None
        self.chat_template = self._load_chat_template(chat_template)
        if _has_transformers and AutoProcessor is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    processor_name or model,
                    trust_remote_code=trust_remote_code,
                    **processor_kwargs,
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to load AutoProcessor for {processor_name or model}: {type(e).__name__}: {e}. "
                    "Falling back to plain-text prompts.",
                    stacklevel=2,
                )
        if self.chat_template is not None and self.processor is not None:
            self.processor.chat_template = self.chat_template

        self.client = Omni(
            model=self.model,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        atexit.register(self.close)

    @staticmethod
    def _maybe_parse_json_dict(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
            return json.loads(value)
        raise TypeError(f"Expected a dict or JSON object string, got {type(value).__name__}")

    @staticmethod
    def _maybe_parse_json_like_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(kwargs)
        for key, value in parsed.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        return parsed

    @staticmethod
    def _normalize_output_modalities(value: Optional[str | list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _load_chat_template(chat_template: Optional[str]) -> Optional[str]:
        if chat_template is None:
            return None
        if os.path.sep in chat_template or chat_template.endswith((".jinja", ".jinja2", ".j2")):
            if not os.path.isfile(chat_template):
                raise FileNotFoundError(f"Chat template file not found: {chat_template}")
            with open(chat_template, "r", encoding="utf-8") as handle:
                return handle.read()
        return chat_template

    def _select_max_new_tokens(self, request_max_new_tokens: Any) -> int:
        if request_max_new_tokens is None:
            return self.max_new_tokens
        try:
            request_max_new_tokens = int(request_max_new_tokens)
        except (TypeError, ValueError):
            return self.max_new_tokens
        return max(request_max_new_tokens, self.max_new_tokens)

    @staticmethod
    def _normalize_top_p_for_vllm(top_p: Any) -> Any:
        if isinstance(top_p, bool):
            return top_p
        try:
            numeric_top_p = float(top_p)
        except (TypeError, ValueError):
            return top_p
        if numeric_top_p == 0.0:
            return 1.0
        return top_p

    def _build_stage0_sampling_params(self, gen_kwargs: dict[str, Any]) -> Sequence[Any]:
        sampling_params_list = copy.deepcopy(list(self.client.default_sampling_params_list))
        if not sampling_params_list:
            return sampling_params_list

        stage0 = sampling_params_list[0]
        gen = dict(gen_kwargs or {})
        diffusion_defaults = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "guidance_scale_2": self.guidance_scale_2,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "fps": self.fps,
            "boundary_ratio": self.boundary_ratio,
            "flow_shift": self.flow_shift,
        }
        for key, value in diffusion_defaults.items():
            if value is not None and hasattr(stage0, key):
                setattr(stage0, key, value)
        if hasattr(stage0, "guidance_scale_provided"):
            setattr(stage0, "guidance_scale_provided", True)
        if hasattr(stage0, "max_tokens"):
            setattr(stage0, "max_tokens", self._select_max_new_tokens(gen.get("max_new_tokens")))
        if hasattr(stage0, "temperature") and "temperature" in gen:
            setattr(stage0, "temperature", gen["temperature"])
        if hasattr(stage0, "top_p") and "top_p" in gen:
            setattr(stage0, "top_p", self._normalize_top_p_for_vllm(gen["top_p"]))

        for key, value in gen.items():
            if key in {"until", "max_new_tokens", "temperature", "top_p"}:
                continue
            if hasattr(stage0, key):
                setattr(stage0, key, value)
        sampling_params_list[0] = stage0
        return sampling_params_list

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        raise AttributeError(f"{type(self.processor).__name__} does not provide apply_chat_template")

    @staticmethod
    def _extract_plain_text_prompt(chat_messages: ChatMessages) -> str:
        texts: list[str] = []
        for msg in chat_messages.messages:
            if msg.role != "user":
                continue
            for content in msg.content:
                if content.type == "text" and content.text:
                    texts.append(content.text)
        return "\n".join(texts).strip()

    @staticmethod
    def _is_video_string(value: Any) -> bool:
        return isinstance(value, str) and value.lower().endswith((".mp4", ".avi", ".mov", ".flv", ".wmv", ".mkv", ".webm"))

    def _video_fetch_kwargs(self) -> dict[str, Any]:
        if self.fps is not None:
            return {"fps": self.fps}
        return {"num_frames": self.nframes}

    @staticmethod
    def _maybe_decode_audio_object(audio_obj: Any) -> tuple[np.ndarray, float] | None:
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            return VLLMOmni._to_numpy_audio(audio_obj["array"]), float(audio_obj.get("sampling_rate", 16000))
        if isinstance(audio_obj, tuple) and len(audio_obj) == 2 and isinstance(audio_obj[1], (int, float)):
            return VLLMOmni._to_numpy_audio(audio_obj[0]), float(audio_obj[1])
        if isinstance(audio_obj, np.ndarray):
            return audio_obj.astype(np.float32, copy=False), 16000.0
        if isinstance(audio_obj, list) and audio_obj and all(isinstance(x, (int, float)) for x in audio_obj):
            return np.asarray(audio_obj, dtype=np.float32), 16000.0
        if torch.is_tensor(audio_obj):
            return audio_obj.detach().cpu().numpy().astype(np.float32, copy=False), 16000.0
        if isinstance(audio_obj, str):
            return None

        candidates = []
        if hasattr(audio_obj, "get_all_samples"):
            try:
                candidates.append(audio_obj.get_all_samples())
            except Exception:
                pass
        if hasattr(audio_obj, "decode"):
            try:
                candidates.append(audio_obj.decode())
            except Exception:
                pass
        if hasattr(audio_obj, "__call__"):
            try:
                candidates.append(audio_obj())
            except Exception:
                pass
        candidates.append(audio_obj)

        for candidate in candidates:
            if isinstance(candidate, dict) and "array" in candidate:
                return VLLMOmni._to_numpy_audio(candidate["array"]), float(candidate.get("sampling_rate", 16000))
            if hasattr(candidate, "array") and hasattr(candidate, "sampling_rate"):
                return VLLMOmni._to_numpy_audio(candidate.array), float(candidate.sampling_rate)
            if hasattr(candidate, "samples"):
                sample_rate = getattr(candidate, "sample_rate", getattr(candidate, "sampling_rate", 16000))
                return VLLMOmni._to_numpy_audio(candidate.samples), float(sample_rate)
            if hasattr(candidate, "data") and hasattr(candidate, "sample_rate"):
                return VLLMOmni._to_numpy_audio(candidate.data), float(candidate.sample_rate)
        return None

    def _prepare_image_input(self, image: Any) -> Any:
        if isinstance(image, (Image.Image, np.ndarray)) or torch.is_tensor(image):
            return image
        if isinstance(image, str):
            if fetch_image is None:
                raise ImportError("vllm.multimodal.utils.fetch_image is required for image path/url inputs.")
            return fetch_image(image)
        return image

    def _prepare_audio_input(self, audio: Any) -> Any:
        decoded = self._maybe_decode_audio_object(audio)
        if decoded is not None:
            return decoded
        if isinstance(audio, dict):
            for key in ("path", "audio", "url"):
                value = audio.get(key)
                if isinstance(value, str):
                    audio = value
                    break
        if isinstance(audio, str):
            if fetch_audio is None:
                raise ImportError("vllm.multimodal.utils.fetch_audio is required for audio path/url inputs.")
            return fetch_audio(audio)
        raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

    def _prepare_video_input(self, video: Any) -> tuple[Any, Any | None]:
        if isinstance(video, tuple) and len(video) == 2 and isinstance(video[1], dict):
            return video, None
        if isinstance(video, np.ndarray) or torch.is_tensor(video):
            return video, None
        if isinstance(video, list) and video:
            return video, None
        if isinstance(video, str):
            if fetch_video is None:
                raise ImportError("vllm.multimodal.utils.fetch_video is required for video path/url inputs.")
            frames, metadata = fetch_video(video, self._video_fetch_kwargs())
            extracted_audio = None
            if self.extract_audio_from_video:
                try:
                    extracted_audio = self._prepare_audio_input(video)
                except Exception:
                    extracted_audio = None
            return (frames, metadata), extracted_audio
        raise TypeError(f"Unsupported video input type: {type(video).__name__}")

    def _build_multi_modal_data(self, chat_messages: ChatMessages) -> dict[str, Any]:
        images, videos, audios = chat_messages.extract_media()
        multi_modal_data: dict[str, Any] = {}

        if images:
            prepared_images = [self._prepare_image_input(image) for image in images]
            if self.processor is None and len(prepared_images) == 1:
                multi_modal_data["image"] = prepared_images[0]
            else:
                multi_modal_data["image"] = prepared_images

        extracted_video_audios = []
        if videos:
            prepared_videos = []
            for video in videos:
                prepared_video, extracted_audio = self._prepare_video_input(video)
                prepared_videos.append(prepared_video)
                if extracted_audio is not None:
                    extracted_video_audios.append(extracted_audio)
            multi_modal_data["video"] = prepared_videos

        all_audios = list(audios) + extracted_video_audios
        if all_audios:
            multi_modal_data["audio"] = [self._prepare_audio_input(audio) for audio in all_audios]

        return multi_modal_data

    def make_one_request(self, request: Instance) -> _PreparedRequest:
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        if self.processor is not None:
            hf_messages = chat_messages.to_hf_messages()
            prompt_text = self._apply_chat_template(hf_messages)
        else:
            prompt_text = self._extract_plain_text_prompt(chat_messages)
        prompt = {"prompt": prompt_text}

        multi_modal_data = self._build_multi_modal_data(chat_messages)
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data
        if self.output_modalities is not None:
            prompt["modalities"] = self.output_modalities

        sampling_params_list = self._build_stage0_sampling_params(dict(gen_kwargs or {}))
        return _PreparedRequest(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            task=str(task),
            split=split,
            doc_id=doc_id,
        )

    @staticmethod
    def _sampling_signature(sampling_params_list: Sequence[Any]) -> tuple[str, ...]:
        return tuple(repr(params) for params in sampling_params_list)

    @staticmethod
    def _extract_text(output: Any) -> str:
        outputs = getattr(output, "outputs", []) or []
        if outputs:
            return getattr(outputs[0], "text", "") or ""
        return ""

    @staticmethod
    def _extract_token_counts(output: Any) -> TokenCounts | None:
        outputs = getattr(output, "outputs", []) or []
        if not outputs:
            return None
        token_ids = getattr(outputs[0], "token_ids", None)
        if token_ids is None:
            return None
        return TokenCounts(output_tokens=len(token_ids))

    @staticmethod
    def _to_numpy_audio(audio: Any) -> np.ndarray:
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        elif isinstance(audio, list):
            audio = np.asarray(audio, dtype=np.float32)
        elif not isinstance(audio, np.ndarray):
            raise TypeError(f"Unsupported audio array type: {type(audio).__name__}")

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > 8:
            audio = audio.T
        return audio

    def _collect_audio_payloads(self, payload: Any, fallback_sr: Optional[float] = None) -> list[tuple[np.ndarray, float]]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            next_sr = payload.get("audio_sample_rate", payload.get("sampling_rate", payload.get("sample_rate", payload.get("sr", fallback_sr))))
            if "audio" in payload:
                return self._collect_audio_payloads(payload["audio"], next_sr)
            if "array" in payload:
                return [(self._to_numpy_audio(payload["array"]), float(next_sr or 16000))]
            clips: list[tuple[np.ndarray, float]] = []
            for value in payload.values():
                clips.extend(self._collect_audio_payloads(value, next_sr))
            return clips
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], (int, float)):
            return [(self._to_numpy_audio(payload[0]), float(payload[1]))]
        if isinstance(payload, (list, tuple)):
            if payload and all(isinstance(item, (int, float)) for item in payload):
                return [(self._to_numpy_audio(payload), float(fallback_sr or 16000))]
            clips: list[tuple[np.ndarray, float]] = []
            for item in payload:
                clips.extend(self._collect_audio_payloads(item, fallback_sr))
            return clips
        if torch.is_tensor(payload) or isinstance(payload, np.ndarray):
            return [(self._to_numpy_audio(payload), float(fallback_sr or 16000))]
        return []

    @staticmethod
    def _to_pil_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image output type: {type(image).__name__}")
        if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1) * 255 if image.max() <= 1.0 else np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        return Image.fromarray(image)

    def _request_output_dir(self, task: str, split: Any, doc_id: Any) -> str:
        out_dir = os.path.join(self.output_dir, _safe(task), _safe(split), _safe(doc_id))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_images(self, images: Sequence[Any], out_dir: str) -> list[str]:
        paths = []
        for idx, image in enumerate(images):
            image_path = os.path.join(out_dir, f"image_{idx}.png")
            self._to_pil_image(image).save(image_path)
            paths.append(image_path)
        return paths

    def _normalize_video_frames(self, frames: Any) -> list[Any]:
        if isinstance(frames, list):
            normalized: list[Any] = []
            for item in frames:
                normalized.extend(self._normalize_video_frames(item))
            return normalized
        if torch.is_tensor(frames):
            frames = frames.detach().cpu().numpy()
        if isinstance(frames, np.ndarray):
            if frames.ndim == 5 and frames.shape[0] == 1:
                return self._normalize_video_frames(frames[0])
            if frames.ndim == 4:
                return [frames[i] for i in range(frames.shape[0])]
            if frames.ndim == 3:
                return [frames]
        return [frames]

    def _save_video(self, images: Sequence[Any], out_dir: str) -> list[str]:
        if not images:
            return []

        video_path = os.path.join(out_dir, "video.mp4")
        fps = int(self.fps or 16)
        pil_images = [self._to_pil_image(image).convert("RGB") for image in self._normalize_video_frames(list(images))]
        if export_to_video is not None:
            export_to_video(pil_images, output_video_path=video_path, fps=fps)
            return [video_path]

        try:
            imageio_v2 = importlib.import_module("imageio.v2")
        except Exception as e:
            raise ImportError("Saving video outputs requires `diffusers` or `imageio`.") from e

        frames = [np.asarray(self._to_pil_image(image).convert("RGB")) for image in images]
        imageio_v2.mimsave(video_path, frames, fps=fps)
        return [video_path]

    def _save_audios(self, audio_payload: Any, out_dir: str, fallback_sr: Optional[float]) -> list[str]:
        clips = self._collect_audio_payloads(audio_payload, fallback_sr=fallback_sr)
        if not clips:
            return []
        if not _has_soundfile or soundfile is None:
            raise ImportError("soundfile is required to save audio outputs from vllm_omni.")

        paths = []
        for idx, (audio, sample_rate) in enumerate(clips):
            audio_path = os.path.join(out_dir, f"audio_{idx}.wav")
            soundfile.write(audio_path, audio, int(round(sample_rate)))
            paths.append(audio_path)
        return paths

    def _format_output(self, text: str, image_paths: list[str], audio_paths: list[str], video_paths: list[str]) -> str:
        if not image_paths and not audio_paths and not video_paths:
            return text
        payload: dict[str, Any] = {"text": text}
        if video_paths:
            payload["videos"] = video_paths
        if image_paths:
            payload["images"] = image_paths
        if audio_paths:
            payload["audios"] = audio_paths
        return json.dumps(payload, ensure_ascii=False)

    def _to_generation_result(self, output: Any, prepared: _PreparedRequest) -> GenerationResult:
        text = self._extract_text(output)
        token_counts = self._extract_token_counts(output)

        image_paths: list[str] = []
        audio_paths: list[str] = []
        video_paths: list[str] = []
        out_dir = self._request_output_dir(prepared.task, prepared.split, prepared.doc_id)

        images = getattr(output, "images", []) or []
        if images:
            if len(images) > 1:
                video_paths = self._save_video(images, out_dir)
            else:
                image_paths = self._save_images(images, out_dir)

        multimodal_output = getattr(output, "multimodal_output", {}) or {}
        fallback_sr = multimodal_output.get("audio_sample_rate", multimodal_output.get("sampling_rate", multimodal_output.get("sample_rate", multimodal_output.get("sr"))))
        if "audio" in multimodal_output:
            audio_paths = self._save_audios(multimodal_output["audio"], out_dir, fallback_sr)

        formatted = self._format_output(text, image_paths, audio_paths, video_paths)
        return GenerationResult(text=formatted, token_counts=token_counts)

    def _generate_batch(self, prepared_requests: Sequence[_PreparedRequest]) -> tuple[list[Any], float]:
        prompts = [prepared.prompt for prepared in prepared_requests]
        start_time = time.time()
        outputs = self.client.generate(
            prompts,
            sampling_params_list=prepared_requests[0].sampling_params_list,
            use_tqdm=False,
        )
        return outputs, time.time() - start_time

    def _generate_single(self, prepared_request: _PreparedRequest) -> tuple[Any, float]:
        start_time = time.time()
        outputs = self.client.generate(
            prepared_request.prompt,
            sampling_params_list=prepared_request.sampling_params_list,
            use_tqdm=False,
        )
        return outputs[0], time.time() - start_time

    def generate_until(self, requests) -> List[GenerationResult]:
        res: list[GenerationResult] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0.0
        total_tokens = 0

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(batch_requests)))) as executor:
                prepared_requests = list(executor.map(self.make_one_request, batch_requests))

            can_batch = len({self._sampling_signature(prepared.sampling_params_list) for prepared in prepared_requests}) == 1
            if can_batch:
                outputs, elapsed = self._generate_batch(prepared_requests)
                total_elapsed_time += elapsed
                for prepared, output in zip(prepared_requests, outputs):
                    result = self._to_generation_result(output, prepared)
                    if result.token_counts and result.token_counts.output_tokens is not None:
                        total_tokens += result.token_counts.output_tokens
                    res.append(result)
            else:
                for prepared in prepared_requests:
                    output, elapsed = self._generate_single(prepared)
                    total_elapsed_time += elapsed
                    result = self._to_generation_result(output, prepared)
                    if result.token_counts and result.token_counts.output_tokens is not None:
                        total_tokens += result.token_counts.output_tokens
                    res.append(result)

            pbar.update(len(batch_requests))

        if not self.disable_log_stats:
            avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0.0
            log_metrics(
                total_elapsed_time=total_elapsed_time,
                total_gen_tokens=total_tokens,
                avg_speed=avg_speed,
                additional_metrics={"request_count": len(requests)},
            )

        pbar.close()
        return res

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "vllm_omni does not support loglikelihood"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

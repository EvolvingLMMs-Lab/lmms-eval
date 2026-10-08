import inspect
import math
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import av
import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

from loguru import logger as eval_logger
from packaging.version import Version

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.model_utils.grt.frozen_video_cache import (
    FrozenDecordFrameCache,
    resize_video_frames,
)
from lmms_eval.models.model_utils.grt.load_video import read_video_pyav_seek_uniform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

PROMPT_ROUTER_OFF = "off"
DENSEVIDEO_LPM_PROMPT_ROUTER = "densevideo_lpm_first_sentence_v1"
DENSEVIDEO_LPM_REFERENCE_FRAMES = 8
VIDEO_DECODE_BACKENDS = {"pyav_seek", "decord_legacy", "decord_frozen"}

GATE_PROJECTION_NATIVE_ON_FULL = "native_on_full"
GATE_PROJECTION_LINEAR_CONSISTENT = "linear_consistent"
GATE_PROJECTION_MODES = {
    GATE_PROJECTION_NATIVE_ON_FULL,
    GATE_PROJECTION_LINEAR_CONSISTENT,
}

_SUBTITLE_QUESTION_PREFIX = re.compile(
    r"^what\s+subtitles\s+appear\s+in\s+the\s+entire\s+video(?:\s|$)",
    flags=re.IGNORECASE,
)
_OCR_QUESTION_PREFIX = re.compile(
    r"^what\s+text\s+is\s+extracted\s+by\s+ocr\s+in\s+the\s+entire\s+video(?:\s|$)",
    flags=re.IGNORECASE,
)


def normalize_gate_projection_mode(value: object) -> str:
    """Normalize the explicit patch-projection kernel policy."""

    mode = str(value or GATE_PROJECTION_NATIVE_ON_FULL).strip().lower()
    if mode not in GATE_PROJECTION_MODES:
        expected = "|".join(sorted(GATE_PROJECTION_MODES))
        raise ValueError(f"Unsupported gate_projection_mode={value}. Expected {expected}.")
    return mode


def normalize_gate_refresh_interval_frames(value: object) -> int:
    """Return a non-negative integral full-refresh interval."""

    if value in (None, "", "none", "None"):
        return 0
    if isinstance(value, bool):
        raise ValueError("gate_refresh_interval_frames must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gate_refresh_interval_frames must be a non-negative integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError("gate_refresh_interval_frames must be a non-negative integer")
    return int(numeric)


@dataclass(frozen=True)
class VideoRequestPlan:
    """Public, prompt-derived execution plan for one video request."""

    prompt_router: str
    question_route: str
    requested_frames: int
    reference_frames: int
    gate_policy: str
    video_decode_backend: str
    max_new_tokens: int


def normalize_video_decode_backend(value: object) -> str:
    backend = str(value or "pyav_seek").strip().lower()
    if backend not in VIDEO_DECODE_BACKENDS:
        expected = "|".join(sorted(VIDEO_DECODE_BACKENDS))
        raise ValueError(f"Unsupported video_decode_backend={value}. Expected {expected}.")
    return backend


def normalize_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be positive integer")
    return int(numeric)


def classify_densevideo_lpm_first_sentence(context: str) -> str:
    """Classify a DIVE LPM query using only its public first sentence.

    This deliberately has no access to the task name, document metadata,
    reference answer, or sample id.  An unfamiliar prompt fails closed to the
    ``unknown`` route, which uses the exact eight-frame path below.
    """

    first_sentence = re.split(r"[.!?]", str(context or ""), maxsplit=1)[0]
    first_sentence = " ".join(first_sentence.strip().split())
    if _SUBTITLE_QUESTION_PREFIX.match(first_sentence):
        return "subtitle"
    if _OCR_QUESTION_PREFIX.match(first_sentence):
        return "ocr"
    return "unknown"


def resolve_video_request_plan(
    context: str,
    *,
    prompt_router: str,
    max_frames_num: int,
    ocr_max_frames_num: int,
    gate_policy: str,
    video_decode_backend: str = "pyav_seek",
    default_max_new_tokens: int = 1024,
    subtitle_video_decode_backend: Optional[str] = None,
    subtitle_max_new_tokens: Optional[int] = None,
) -> VideoRequestPlan:
    """Resolve public route controls without consulting hidden labels."""

    router = str(prompt_router or PROMPT_ROUTER_OFF).strip().lower()
    if router not in {PROMPT_ROUTER_OFF, DENSEVIDEO_LPM_PROMPT_ROUTER}:
        raise ValueError(f"Unsupported prompt_router={prompt_router}. Expected {PROMPT_ROUTER_OFF}|{DENSEVIDEO_LPM_PROMPT_ROUTER}.")
    default_frames = normalize_positive_integer(max_frames_num, name="max_frames_num")
    ocr_frames = normalize_positive_integer(ocr_max_frames_num, name="ocr_max_frames_num")
    default_token_cap = normalize_positive_integer(default_max_new_tokens, name="default_max_new_tokens")
    default_backend = normalize_video_decode_backend(video_decode_backend)
    subtitle_backend = default_backend if subtitle_video_decode_backend in (None, "", "none", "None") else normalize_video_decode_backend(subtitle_video_decode_backend)
    subtitle_token_cap = default_token_cap if subtitle_max_new_tokens in (None, "", "none", "None") else normalize_positive_integer(subtitle_max_new_tokens, name="subtitle_max_new_tokens")
    configured_gate_policy = str(gate_policy or "motion").strip().lower()
    if configured_gate_policy not in {"motion", "all"}:
        raise ValueError(f"Unsupported gate_policy={gate_policy}. Expected motion|all.")

    if router == PROMPT_ROUTER_OFF:
        if subtitle_video_decode_backend not in (None, "", "none", "None"):
            raise ValueError(f"subtitle_video_decode_backend requires prompt_router={DENSEVIDEO_LPM_PROMPT_ROUTER}")
        if subtitle_max_new_tokens not in (None, "", "none", "None"):
            raise ValueError(f"subtitle_max_new_tokens requires prompt_router={DENSEVIDEO_LPM_PROMPT_ROUTER}")
        return VideoRequestPlan(
            prompt_router=router,
            question_route="unrouted",
            requested_frames=default_frames,
            reference_frames=default_frames,
            gate_policy=configured_gate_policy,
            video_decode_backend=default_backend,
            max_new_tokens=default_token_cap,
        )

    question_route = classify_densevideo_lpm_first_sentence(context)
    if question_route == "ocr":
        return VideoRequestPlan(
            prompt_router=router,
            question_route=question_route,
            requested_frames=ocr_frames,
            reference_frames=DENSEVIDEO_LPM_REFERENCE_FRAMES,
            gate_policy=configured_gate_policy,
            video_decode_backend=default_backend,
            max_new_tokens=default_token_cap,
        )
    if question_route == "subtitle":
        return VideoRequestPlan(
            prompt_router=router,
            question_route=question_route,
            requested_frames=DENSEVIDEO_LPM_REFERENCE_FRAMES,
            reference_frames=DENSEVIDEO_LPM_REFERENCE_FRAMES,
            gate_policy="all",
            video_decode_backend=subtitle_backend,
            max_new_tokens=subtitle_token_cap,
        )
    return VideoRequestPlan(
        prompt_router=router,
        question_route=question_route,
        requested_frames=DENSEVIDEO_LPM_REFERENCE_FRAMES,
        reference_frames=DENSEVIDEO_LPM_REFERENCE_FRAMES,
        gate_policy="all",
        video_decode_backend=default_backend,
        max_new_tokens=default_token_cap,
    )


# Default chat for llava-hf/llava-1.5 models: https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"

model_map = {
    "llava": LlavaForConditionalGeneration,
    "llava_next": LlavaNextForConditionalGeneration,
}

try:
    from transformers import LlavaOnevisionForConditionalGeneration

    model_map["llava_onevision"] = LlavaOnevisionForConditionalGeneration
except Exception:
    eval_logger.debug("Transformers version does not support llava-onevision. Skipping.")


def _model_class_from_config(config):
    model_type = getattr(config, "model_type", "llava")
    if model_type in model_map:
        return model_map[model_type]
    auto_map = getattr(config, "auto_map", {}) or {}
    if "AutoModelForCausalLM" in auto_map:
        eval_logger.info(f"Using AutoModelForCausalLM for custom model_type={model_type}")
        return AutoModelForCausalLM
    for auto_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        if auto_cls is not None:
            eval_logger.info(f"Using {auto_cls.__name__} for custom model_type={model_type}")
            return auto_cls
    raise KeyError(f"Unsupported model_type={model_type!r}. Upgrade transformers or add a model class mapping for this checkpoint.")


def _patch_flash_attention_varlen_func() -> None:
    try:
        import transformers.modeling_flash_attention_utils as flash_utils
    except Exception:
        return
    if hasattr(flash_utils, "flash_attn_varlen_func"):
        return
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception:

        def flash_attn_varlen_func(*args, **kwargs):
            raise ImportError("flash_attn_varlen_func requires flash-attn; retry with attn_implementation=eager.")

    flash_utils.flash_attn_varlen_func = flash_attn_varlen_func


def _patch_transformers_config_compat() -> None:
    try:
        import transformers.configuration_utils as config_utils
    except Exception:
        return
    if not hasattr(config_utils, "PreTrainedConfig") and hasattr(config_utils, "PretrainedConfig"):
        config_utils.PreTrainedConfig = config_utils.PretrainedConfig


class GatedSiglipVisionEmbeddings(torch.nn.Module):
    """Drop-in SigLIP embeddings with frame-to-frame patch reuse for video.

    Hugging Face's LLaVA-OneVision flattens ``[batch, frames, C, H, W]`` to
    ``[batch * frames, C, H, W]`` immediately before calling the SigLIP vision
    tower.  ``set_video_frames`` supplies the missing frame dimension for that
    call.  With no active video (images), ``gate_policy=all``, or a negative
    threshold, this module takes the native Conv2d path.

    Gating only skips patch-embedding projections.  The returned tensor keeps
    every visual token, so the rest of the official HF OneVision model remains
    unchanged.
    """

    def __init__(
        self,
        original_embeddings,
        *,
        diff_threshold: float = 0.01,
        gate_policy: str = "motion",
        gate_metric: str = "ssim",
        gate_projection_mode: str = GATE_PROJECTION_NATIVE_ON_FULL,
        gate_refresh_interval_frames: int = 0,
        parent=None,
    ) -> None:
        super().__init__()
        required = ("patch_embedding", "position_embedding", "position_ids", "patch_size")
        missing = [name for name in required if not hasattr(original_embeddings, name)]
        if missing:
            raise TypeError(f"Unsupported SigLIP embeddings; missing {', '.join(missing)}")

        self.parent = parent
        self.config = getattr(original_embeddings, "config", None)
        self.embed_dim = int(getattr(original_embeddings, "embed_dim", original_embeddings.patch_embedding.out_channels))
        self.image_size = getattr(original_embeddings, "image_size", None)
        self.patch_size = getattr(original_embeddings, "patch_size")
        self.num_patches = int(getattr(original_embeddings, "num_patches", original_embeddings.position_embedding.num_embeddings))
        self.num_positions = int(getattr(original_embeddings, "num_positions", self.num_patches))
        self.patch_embedding = original_embeddings.patch_embedding
        self.position_embedding = original_embeddings.position_embedding
        self.register_buffer("position_ids", original_embeddings.position_ids.detach().clone(), persistent=False)

        self.diff_threshold = float(diff_threshold)
        self.gate_policy = str(gate_policy or "motion").strip().lower()
        if self.gate_policy not in {"motion", "all"}:
            raise ValueError(f"Unsupported gate_policy={gate_policy}. Expected motion|all.")
        self.gate_metric = str(gate_metric or "ssim").strip().lower()
        if self.gate_metric not in {"ssim", "l2"}:
            raise ValueError(f"Unsupported gate_metric={gate_metric}. Expected ssim|l2.")
        self.gate_projection_mode = normalize_gate_projection_mode(gate_projection_mode)
        self.gate_refresh_interval_frames = normalize_gate_refresh_interval_frames(gate_refresh_interval_frames)

        self._active_video_frames = None
        self._active_gate_policy = None
        self.last_keep_flat = None
        self.last_diff_score_flat = None
        self.last_forced_refresh_frame_indices = ()

    def set_video_frames(self, num_frames: Optional[int]) -> None:
        """Mark subsequent forwards as video, or pass ``None`` for image calls."""

        self.set_video_context(num_frames, gate_policy=None)

    def set_video_context(self, num_frames: Optional[int], gate_policy: Optional[str] = None) -> None:
        """Set request-local video state without mutating the configured policy."""

        if num_frames is None:
            self._active_video_frames = None
            self._active_gate_policy = None
            return
        num_frames = int(num_frames)
        self._active_video_frames = num_frames if num_frames > 0 else None
        if self._active_video_frames is None or gate_policy is None:
            self._active_gate_policy = None
            return
        active_policy = str(gate_policy).strip().lower()
        if active_policy not in {"motion", "all"}:
            raise ValueError(f"Unsupported active gate_policy={gate_policy}. Expected motion|all.")
        self._active_gate_policy = active_policy

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Match ``transformers.SiglipVisionEmbeddings`` interpolation."""

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids.to(device=self.position_embedding.weight.device))

        position_embedding = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]
        patch_height = self.patch_embedding.kernel_size[0]
        patch_width = self.patch_embedding.kernel_size[1]
        new_height = height // patch_height
        new_width = width // patch_width
        sqrt_num_positions = int(num_positions**0.5)
        if sqrt_num_positions * sqrt_num_positions != num_positions:
            raise ValueError(f"Cannot interpolate non-square SigLIP position grid with {num_positions} positions")
        position_embedding = position_embedding.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        position_embedding = position_embedding.permute(0, 3, 1, 2)
        position_embedding = torch.nn.functional.interpolate(
            position_embedding,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        return position_embedding.permute(0, 2, 3, 1).view(1, -1, dim)

    @staticmethod
    def _patch_ssim_dissimilarity(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Per-patch SSIM dissimilarity used by the existing dense wrapper."""

        x = current.float()
        y = previous.float()
        mean_x = x.mean(dim=-1)
        mean_y = y.mean(dim=-1)
        centered_x = x - mean_x.unsqueeze(-1)
        centered_y = y - mean_y.unsqueeze(-1)
        variance_x = (centered_x * centered_x).mean(dim=-1)
        variance_y = (centered_y * centered_y).mean(dim=-1)
        covariance = (centered_x * centered_y).mean(dim=-1)
        dynamic_range = torch.maximum(
            x.amax(dim=-1) - x.amin(dim=-1),
            y.amax(dim=-1) - y.amin(dim=-1),
        ).clamp_min(1.0)
        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2
        numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
        denominator = (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
        ssim = numerator / denominator.clamp_min(1e-12)
        return (1.0 - ssim).clamp(min=0.0, max=2.0)

    def _add_position_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        interpolate_pos_encoding: bool,
    ) -> torch.Tensor:
        if interpolate_pos_encoding:
            position = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position = self.position_embedding(self.position_ids.to(device=self.position_embedding.weight.device))
        return embeddings + position.to(dtype=embeddings.dtype, device=embeddings.device)

    def _record_metrics(
        self,
        keep: torch.Tensor,
        score: torch.Tensor,
        gate_policy: str,
        *,
        forced_refresh_frame_indices=(),
    ) -> None:
        self.last_keep_flat = keep.reshape(-1, keep.shape[-1])
        self.last_diff_score_flat = score.reshape(-1, score.shape[-1])
        self.last_forced_refresh_frame_indices = tuple(int(index) for index in forced_refresh_frame_indices)
        if self.parent is None:
            return
        orig_patches = int(keep.numel())
        recomputed_patches = int(keep.sum().item())
        ratio = recomputed_patches / float(orig_patches) if orig_patches else 1.0
        self.parent._last_gate_keep_ratio = float(ratio)
        self.parent._last_recomputed_patches = recomputed_patches
        self.parent._last_orig_patches = orig_patches
        self.parent._last_recompute_ratio = float(ratio)
        self.parent._last_gate_policy = gate_policy
        self.parent._last_gate_metric = self.gate_metric
        self.parent._last_gate_projection_mode = self.gate_projection_mode
        self.parent._last_gate_refresh_interval_frames = self.gate_refresh_interval_frames
        forced_refresh_frames = int(keep.shape[0]) * len(self.last_forced_refresh_frame_indices)
        self.parent._last_forced_refresh_frames = forced_refresh_frames
        self.parent._last_forced_refresh_patches = forced_refresh_frames * int(keep.shape[-1])
        self.parent._last_forced_refresh_frame_indices = ",".join(str(index) for index in self.last_forced_refresh_frame_indices) if self.last_forced_refresh_frame_indices else "none"
        observed_frames = int(keep.shape[1])
        reference_frames = int(getattr(self.parent, "_last_reference_frames", observed_frames) or observed_frames)
        reference_orig_patches = int(keep.shape[0] * reference_frames * keep.shape[-1])
        self.parent._last_reference_orig_patches = reference_orig_patches
        self.parent._last_patch_projection_recompute_ratio = float(ratio)
        self.parent._last_patch_projection_compute_ratio_vs_reference = recomputed_patches / float(reference_orig_patches) if reference_orig_patches else 1.0

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected SigLIP pixel_values with 4 dimensions, got shape={tuple(pixel_values.shape)}")

        flat_batch, channels, height, width = pixel_values.shape
        frames = self._active_video_frames or 1
        if flat_batch == 0 or frames <= 0 or flat_batch % frames:
            frames = 1
        batch = flat_batch // frames
        target_dtype = self.patch_embedding.weight.dtype
        values = pixel_values.to(dtype=target_dtype)
        gate_policy = self._active_gate_policy or self.gate_policy

        # Exact native fast path.  This both preserves image behavior and makes
        # gate_policy=all a true numerical control rather than an approximation.
        if frames == 1 or gate_policy == "all" or self.diff_threshold < 0:
            embeddings = self.patch_embedding(values).flatten(2).transpose(1, 2)
            patches_per_frame = embeddings.shape[1]
            keep = torch.ones(batch, frames, patches_per_frame, dtype=torch.bool, device=values.device)
            score = torch.ones(batch, frames, patches_per_frame, dtype=torch.float32, device=values.device)
            self._record_metrics(keep, score, gate_policy)
            return self._add_position_encoding(embeddings, height, width, interpolate_pos_encoding)

        kernel_size = self.patch_embedding.kernel_size
        stride = self.patch_embedding.stride
        padding = self.patch_embedding.padding
        if isinstance(padding, str):
            if padding.lower() != "valid":
                raise ValueError(f"GRT does not support Conv2d padding={padding!r}")
            padding = 0
        dilation = self.patch_embedding.dilation
        if self.patch_embedding.groups != 1:
            raise ValueError("GRT currently requires an ungrouped SigLIP patch Conv2d")

        patches = (
            torch.nn.functional.unfold(
                values.contiguous(),
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride,
            )
            .transpose(1, 2)
            .contiguous()
        )
        patches_per_frame = patches.shape[1]
        patch_vector_dim = patches.shape[2]
        patches = patches.view(batch, frames, patches_per_frame, patch_vector_dim)

        keep = torch.zeros(batch, frames, patches_per_frame, dtype=torch.bool, device=values.device)
        score = torch.zeros(batch, frames, patches_per_frame, dtype=torch.float32, device=values.device)
        keep[:, 0, :] = True
        score[:, 0, :] = 1.0

        # Compare against the raw patch that produced the currently cached
        # embedding, not merely the previous frame.  Adjacent-frame checks can
        # otherwise accumulate many sub-threshold changes while indefinitely
        # reusing a much older embedding.  Updating this reference only when a
        # patch is recomputed makes the configured threshold a real upper bound
        # on cache-source drift.
        cached_source_patches = patches[:, 0].clone()
        forced_refresh_frame_indices = []
        for frame_index in range(1, frames):
            current_patches = patches[:, frame_index]
            if self.gate_metric == "ssim":
                diffs = self._patch_ssim_dissimilarity(current_patches, cached_source_patches)
            else:
                diffs = (current_patches - cached_source_patches).norm(dim=-1)
            frame_keep = diffs > self.diff_threshold
            if self.gate_refresh_interval_frames > 0 and frame_index % self.gate_refresh_interval_frames == 0:
                frame_keep = torch.ones_like(frame_keep)
                forced_refresh_frame_indices.append(frame_index)
            keep[:, frame_index] = frame_keep
            denominator = diffs.float().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            score[:, frame_index] = diffs.float() / denominator
            cached_source_patches = torch.where(
                frame_keep.unsqueeze(-1),
                current_patches,
                cached_source_patches,
            )

        # Compute each video's first frame with the original Conv2d, then only
        # project selected patches in later frames and reuse the previous result.
        video_values = values.reshape(batch, frames, channels, height, width)
        first_embeddings = self.patch_embedding(video_values[:, 0]).flatten(2).transpose(1, 2)
        embeddings = first_embeddings.new_empty(batch, frames, patches_per_frame, self.embed_dim)
        embeddings[:, 0] = first_embeddings
        weight = self.patch_embedding.weight.flatten(1)
        bias = self.patch_embedding.bias
        for frame_index in range(1, frames):
            current = embeddings[:, frame_index - 1].clone()
            frame_keep = keep[:, frame_index]
            if self.gate_projection_mode == GATE_PROJECTION_NATIVE_ON_FULL and frame_keep.all():
                # Preserve native Conv2d numerics when no patch is reusable.
                # The sparse F.linear path can differ from Conv2d by a few ulps.
                current = self.patch_embedding(video_values[:, frame_index]).flatten(2).transpose(1, 2)
            elif frame_keep.any():
                selected_patches = patches[:, frame_index].reshape(-1, patch_vector_dim)[frame_keep.reshape(-1)]
                selected_embeddings = torch.nn.functional.linear(selected_patches, weight, bias)
                current.reshape(-1, self.embed_dim)[frame_keep.reshape(-1)] = selected_embeddings
            embeddings[:, frame_index] = current

        embeddings = embeddings.reshape(flat_batch, patches_per_frame, self.embed_dim)
        self._record_metrics(
            keep,
            score,
            gate_policy,
            forced_refresh_frame_indices=forced_refresh_frame_indices,
        )
        return self._add_position_encoding(embeddings, height, width, interpolate_pos_encoding)


class LlavaHf(lmms):
    """
    Llava Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    Adapted from the InstructBLIP model in lmms_eval/models/instructblip.py

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        max_image_size: Optional[int] = None,
        profiling: Optional[bool] = False,
        use_gated_tok: Optional[bool] = False,
        gate_diff_threshold: Optional[float] = 0.01,
        gate_policy: Optional[str] = "motion",
        gate_metric: Optional[str] = "ssim",
        gate_projection_mode: Optional[str] = GATE_PROJECTION_NATIVE_ON_FULL,
        gate_refresh_interval_frames: Optional[int] = 0,
        prompt_router: Optional[str] = PROMPT_ROUTER_OFF,
        ocr_max_frames_num: Optional[int] = None,
        video_decode_backend: Optional[str] = "pyav_seek",
        subtitle_video_decode_backend: Optional[str] = None,
        subtitle_max_new_tokens: Optional[int] = None,
        decord_frame_cache_manifest: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Reject misspelled/invalid experimental policies before any config or
        # checkpoint access.  This also makes a queued benchmark fail closed
        # instead of spending minutes loading a model it cannot evaluate.
        normalized_gate_projection_mode = normalize_gate_projection_mode(gate_projection_mode)
        normalized_gate_refresh_interval_frames = normalize_gate_refresh_interval_frames(gate_refresh_interval_frames)

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        if "llava-onevision-2-" in pretrained.lower() and Version(transformers.__version__) < Version("5.7.0"):
            raise RuntimeError(f"{pretrained} requires transformers>=5.7.0 (found {transformers.__version__}). Run this checkpoint with the isolated LLaVA-OneVision-2 environment.")
        _patch_transformers_config_compat()
        config = AutoConfig.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        self.max_frames_num = int(max_frames_num)
        self.ocr_max_frames_num = int(ocr_max_frames_num if ocr_max_frames_num not in (None, "", "none", "None") else DENSEVIDEO_LPM_REFERENCE_FRAMES)
        self.max_image_size = None if max_image_size in (None, "", "none", "None") else int(max_image_size)
        if isinstance(profiling, str):
            self.profiling = profiling.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            self.profiling = bool(profiling)
        self._last_sampled_frames = 0
        self._last_video_duration = 0.0
        self._last_effective_fps = 0.0
        if isinstance(use_gated_tok, str):
            self.use_gated_tok = use_gated_tok.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            self.use_gated_tok = bool(use_gated_tok)
        self.gate_diff_threshold = float(gate_diff_threshold)
        self.gate_policy = str(gate_policy or "motion").strip().lower()
        self.gate_metric = str(gate_metric or "ssim").strip().lower()
        self.gate_projection_mode = normalized_gate_projection_mode
        self.gate_refresh_interval_frames = normalized_gate_refresh_interval_frames
        self.prompt_router = str(prompt_router or PROMPT_ROUTER_OFF).strip().lower()
        self.video_decode_backend = normalize_video_decode_backend(video_decode_backend)
        self.subtitle_video_decode_backend = None if subtitle_video_decode_backend in (None, "", "none", "None") else normalize_video_decode_backend(subtitle_video_decode_backend)
        self.subtitle_max_new_tokens = None if subtitle_max_new_tokens in (None, "", "none", "None") else normalize_positive_integer(subtitle_max_new_tokens, name="subtitle_max_new_tokens")
        self.decord_frame_cache_manifest = None if decord_frame_cache_manifest in (None, "", "none", "None") else str(decord_frame_cache_manifest)
        # Validate all request-plan arguments before the heavyweight model load.
        resolve_video_request_plan(
            "",
            prompt_router=self.prompt_router,
            max_frames_num=self.max_frames_num,
            ocr_max_frames_num=self.ocr_max_frames_num,
            gate_policy=self.gate_policy,
            video_decode_backend=self.video_decode_backend,
            default_max_new_tokens=1024,
            subtitle_video_decode_backend=self.subtitle_video_decode_backend,
            subtitle_max_new_tokens=self.subtitle_max_new_tokens,
        )
        frozen_backends = {
            backend
            for backend in (
                self.video_decode_backend,
                self.subtitle_video_decode_backend,
            )
            if backend == "decord_frozen"
        }
        if self.video_decode_backend == "decord_frozen" and self.prompt_router != PROMPT_ROUTER_OFF:
            raise ValueError("video_decode_backend=decord_frozen is only supported with prompt_router=off; use subtitle_video_decode_backend=decord_frozen for the routed subtitle cache")
        if bool(frozen_backends) != bool(self.decord_frame_cache_manifest):
            raise ValueError("decord_frame_cache_manifest must be provided exactly when a decord_frozen backend is configured")
        self._frozen_decord_cache = None
        if frozen_backends:
            frozen_frames = self.max_frames_num if self.video_decode_backend == "decord_frozen" else DENSEVIDEO_LPM_REFERENCE_FRAMES
            self._frozen_decord_cache = FrozenDecordFrameCache(
                self.decord_frame_cache_manifest,
                requested_frames=frozen_frames,
                max_image_size=self.max_image_size,
            )
        self._last_gate_keep_ratio = None
        self._last_recomputed_patches = None
        self._last_orig_patches = None
        self._last_recompute_ratio = None
        self._last_gate_policy = "disabled"
        self._last_gate_metric = "none"
        self._last_gate_projection_mode = self.gate_projection_mode if self.use_gated_tok else "disabled"
        self._last_gate_refresh_interval_frames = self.gate_refresh_interval_frames if self.use_gated_tok else 0
        self._last_forced_refresh_frames = 0
        self._last_forced_refresh_patches = 0
        self._last_forced_refresh_frame_indices = "none"
        self._last_prompt_router = self.prompt_router
        self._last_question_route = "unrouted"
        self._last_video_decode_backend = self.video_decode_backend
        self._last_effective_max_new_tokens = 1024
        self._last_frozen_cache_hit = False
        self._last_frozen_cache_manifest_sha256 = self._frozen_decord_cache.manifest_sha256 if self._frozen_decord_cache is not None else "none"
        self._last_frozen_cache_array_sha256 = "none"
        self._last_frozen_cache_video_key = "none"
        self._last_requested_frames = self.max_frames_num
        self._last_reference_frames = self.max_frames_num
        self._last_reference_orig_patches = None
        self._last_patch_projection_recompute_ratio = None
        self._last_patch_projection_compute_ratio_vs_reference = None
        self._gated_vision_embeddings = None
        if self.use_gated_tok and getattr(config, "model_type", None) != "llava_onevision":
            raise ValueError("use_gated_tok is supported only for official Hugging Face LlavaOnevision models")
        model_type = _model_class_from_config(config)
        _patch_flash_attention_varlen_func()
        self._model = model_type.from_pretrained(pretrained, revision=revision, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation)
        if self.use_gated_tok:
            self._install_gated_siglip_embeddings()

        self.pretrained = pretrained
        processor_kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if "llava-onevision-1.5" in pretrained.lower():
            processor_kwargs["fix_mistral_regex"] = True
        self._image_processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
        if self._processor_owns_video_decode() and self.max_image_size:
            # OneVision-2 samples file paths inside its custom processor.  Its
            # frame path reads this attribute (the generic ``max_pixels`` call
            # kwarg is reserved for the optional codec backend).
            self._image_processor.video_processor.max_pixels = self.max_image_size**2
        # Pad from left for batched generation: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        self._image_processor.tokenizer.padding_side = "left"
        self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) is str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
            image_tokens = " ".join(image_tokens)
            context = f"{image_tokens}\n{context}"
            # Apply chat template
            messages = [{"role": "user", "content": context}, {"role": "assistant", "content": continuation}]
            if self.chat_template is not None:
                self.tokenizer.chat_template = self.chat_template
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            elif self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prompt_and_continuation = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            formatted_contexts = [prompt]
            formatted_continuation = [prompt_and_continuation]
            model_inputs = self._image_processor(text=formatted_continuation, images=visuals, return_tensors="pt").to(self._device, self.model.dtype)
            labels = model_inputs["input_ids"].clone()
            contxt_id = self._image_processor(text=formatted_contexts, return_tensors="pt")["input_ids"]
            labels[:, : contxt_id.shape[1]] = -100

            if self.accelerator.is_main_process and doc_id % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id}:\n\n{formatted_contexts[0]}\n")
                eval_logger.debug(f"Prompt and continuation for doc ID {doc_id}:\n\n{formatted_continuation[0]}\n")

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = model_inputs["input_ids"][:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : model_inputs["input_ids"].shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num, *, video_decode_backend=None):
        path = video_path if isinstance(video_path, str) else video_path[0]
        requested_frames = int(max_frames_num)
        if requested_frames <= 0:
            raise ValueError(f"max_frames_num must be positive, got {max_frames_num}")
        backend = normalize_video_decode_backend(self.video_decode_backend if video_decode_backend in (None, "", "none", "None") else video_decode_backend)
        self._last_video_decode_backend = backend
        self._last_frozen_cache_hit = False
        self._last_frozen_cache_array_sha256 = "none"
        self._last_frozen_cache_video_key = "none"

        if backend == "pyav_seek":
            with av.open(path) as container:
                stream = container.streams.video[0]
                orig_fps = float(stream.average_rate) if stream.average_rate else 0.0
                total_frame_num = int(stream.frames or 0)
                if total_frame_num <= 0 and stream.duration and stream.time_base and orig_fps > 0:
                    total_frame_num = max(1, int(float(stream.duration * stream.time_base) * orig_fps))
            if total_frame_num <= 0:
                raise ValueError(f"Empty video: {video_path}")
            sample_count = min(requested_frames, total_frame_num)
            spare_frames = read_video_pyav_seek_uniform(path, num_frm=sample_count)
            actual_reference_frames = min(int(self._last_reference_frames), total_frame_num)
        elif backend == "decord_legacy":
            # Preserve the original llava_hf sampling behavior for published
            # artifact reproduction.  In particular, np.linspace intentionally
            # repeats indices when a video has fewer frames than the request.
            from decord import VideoReader, cpu

            video_reader = VideoReader(path, ctx=cpu(0))
            total_frame_num = len(video_reader)
            if total_frame_num <= 0:
                raise ValueError(f"Empty video: {video_path}")
            orig_fps = float(video_reader.get_avg_fps() or 0.0)
            frame_indices = np.linspace(0, total_frame_num - 1, requested_frames, dtype=int).tolist()
            spare_frames = video_reader.get_batch(frame_indices).asnumpy()
            actual_reference_frames = int(self._last_reference_frames)
        else:
            if self._frozen_decord_cache is None:
                raise RuntimeError("decord_frozen was selected without a validated frame cache")
            frozen = self._frozen_decord_cache.load(path)
            spare_frames = frozen.frames
            if spare_frames.shape[0] != requested_frames:
                raise ValueError(f"Frozen Decord frame-count mismatch: cache={spare_frames.shape[0]}, request={requested_frames}")
            total_frame_num = frozen.total_frames
            orig_fps = frozen.avg_fps
            actual_reference_frames = int(self._last_reference_frames)
            self._last_frozen_cache_hit = True
            self._last_frozen_cache_array_sha256 = frozen.array_sha256
            self._last_frozen_cache_video_key = frozen.video_key

        duration_sec = total_frame_num / orig_fps if orig_fps > 0 else 0.0
        self._last_sampled_frames = int(spare_frames.shape[0])
        self._last_reference_frames = actual_reference_frames
        self._last_video_duration = duration_sec
        self._last_effective_fps = self._last_sampled_frames / duration_sec if duration_sec > 0 else 0.0
        if self.profiling:
            print(
                f"[FPS_STATS] strategy=uniform decoder={backend} "
                f"target_fps=none orig_fps={orig_fps:.6f} duration_s={duration_sec:.6f} "
                f"total_frames={total_frame_num} capped_frames={total_frame_num} "
                f"sampled_frames={self._last_sampled_frames} "
                f"requested_frames={self._last_requested_frames} reference_frames={self._last_reference_frames} "
                f"prompt_router={self._last_prompt_router} question_route={self._last_question_route} "
                f"frozen_cache_hit={str(self._last_frozen_cache_hit).lower()} "
                f"frozen_cache_manifest_sha256={self._last_frozen_cache_manifest_sha256} "
                f"frozen_cache_array_sha256={self._last_frozen_cache_array_sha256} "
                f"effective_fps={self._last_effective_fps:.6f}",
                flush=True,
            )
        if backend != "decord_frozen":
            spare_frames = resize_video_frames(spare_frames, self.max_image_size)
        return spare_frames  # (frames, height, width, channels)

    def _input_device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return self._device

    def _install_gated_siglip_embeddings(self) -> None:
        try:
            vision_tower = self._model.model.vision_tower
            vision_model = getattr(vision_tower, "vision_model", vision_tower)
            original_embeddings = vision_model.embeddings
        except AttributeError as exc:
            raise RuntimeError("Could not locate official LLaVA-OneVision SigLIP embeddings for GRT") from exc

        gated_embeddings = GatedSiglipVisionEmbeddings(
            original_embeddings,
            diff_threshold=self.gate_diff_threshold,
            gate_policy=self.gate_policy,
            gate_metric=self.gate_metric,
            gate_projection_mode=self.gate_projection_mode,
            gate_refresh_interval_frames=self.gate_refresh_interval_frames,
            parent=self,
        )
        vision_model.embeddings = gated_embeddings
        self._gated_vision_embeddings = gated_embeddings
        eval_logger.info(
            "Installed official HF LLaVA-OneVision GRT embeddings: "
            f"policy={self.gate_policy}, metric={self.gate_metric}, "
            f"threshold={self.gate_diff_threshold}, "
            f"projection_mode={self.gate_projection_mode}, "
            f"refresh_interval_frames={self.gate_refresh_interval_frames}"
        )

    @staticmethod
    def _video_frame_count(inputs) -> Optional[int]:
        pixel_values = inputs.get("pixel_values_videos") if hasattr(inputs, "get") else None
        if not isinstance(pixel_values, torch.Tensor):
            return None
        if pixel_values.ndim == 5:
            return int(pixel_values.shape[1])
        if pixel_values.ndim == 4:
            return int(pixel_values.shape[0])
        return None

    def _set_gated_video_context(self, task_type: str, inputs, gate_policy: Optional[str] = None) -> None:
        gated_embeddings = self._gated_vision_embeddings
        if gated_embeddings is None:
            return
        if task_type != "video":
            gated_embeddings.set_video_context(None)
            return

        video_frames = self._video_frame_count(inputs)
        pixel_values = inputs.get("pixel_values_videos") if hasattr(inputs, "get") else None
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 4 and self._last_sampled_frames:
            video_frames = int(self._last_sampled_frames)
        if video_frames is None and self._last_sampled_frames:
            video_frames = int(self._last_sampled_frames)
        gated_embeddings.set_video_context(video_frames, gate_policy=gate_policy or self.gate_policy)
        if video_frames is not None and self._last_sampled_frames <= 0:
            self._last_sampled_frames = video_frames
        if video_frames is not None:
            self._last_reference_frames = min(int(self._last_reference_frames), video_frames)

    def _record_ungated_patch_metrics(self, task_type: str, inputs) -> None:
        """Record the native Conv2d patch budget for a non-GRT control."""

        if task_type != "video":
            return
        pixel_values = inputs.get("pixel_values_videos") if hasattr(inputs, "get") else None
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim < 4:
            return
        video_frames = int(self._last_sampled_frames or self._video_frame_count(inputs) or 0)
        if video_frames <= 0:
            return
        try:
            vision_tower = self._model.model.vision_tower
            vision_model = getattr(vision_tower, "vision_model", vision_tower)
            patch_embedding = vision_model.embeddings.patch_embedding
        except AttributeError:
            return

        def pair(value):
            return tuple(value) if isinstance(value, (list, tuple)) else (value, value)

        height, width = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
        kernel_h, kernel_w = pair(patch_embedding.kernel_size)
        stride_h, stride_w = pair(patch_embedding.stride)
        dilation_h, dilation_w = pair(patch_embedding.dilation)
        padding = patch_embedding.padding
        if isinstance(padding, str):
            if padding.lower() == "same":
                out_h = (height + stride_h - 1) // stride_h
                out_w = (width + stride_w - 1) // stride_w
            elif padding.lower() == "valid":
                out_h = (height - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
                out_w = (width - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
            else:
                return
        else:
            padding_h, padding_w = pair(padding)
            out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
            out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        patches_per_frame = max(0, int(out_h) * int(out_w))
        orig_patches = video_frames * patches_per_frame
        reference_orig_patches = int(self._last_reference_frames) * patches_per_frame
        self._last_gate_keep_ratio = 1.0
        self._last_recomputed_patches = orig_patches
        self._last_orig_patches = orig_patches
        self._last_recompute_ratio = 1.0
        self._last_gate_projection_mode = "disabled"
        self._last_gate_refresh_interval_frames = 0
        self._last_forced_refresh_frames = 0
        self._last_forced_refresh_patches = 0
        self._last_forced_refresh_frame_indices = "none"
        self._last_reference_orig_patches = reference_orig_patches
        self._last_patch_projection_recompute_ratio = 1.0
        self._last_patch_projection_compute_ratio_vs_reference = orig_patches / float(reference_orig_patches) if reference_orig_patches else 1.0

    def _processor_owns_video_decode(self) -> bool:
        """The OneVision-2 remote processor expects a path and samples frames itself."""

        processor_cls = type(self._image_processor)
        return processor_cls.__name__ == "LlavaOnevision2Processor" or "llava_onevision2" in processor_cls.__module__.lower()

    def _model_accepts_kwarg(self, name: str) -> bool:
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return True
        return name in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    def _processor_messages(self, task_type: str, context: str):
        """Use structured multimodal chat content when the processor supports it."""

        if task_type == "video":
            content = [{"type": "video"}, {"type": "text", "text": context}]
        elif task_type == "image":
            content = [{"type": "image"}, {"type": "text", "text": context}]
        else:
            content = context
        return [{"role": "user", "content": content}]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            if len(visuals) == 0:
                task_type = "text"
            elif isinstance(visuals[0], PIL.Image.Image):
                task_type = "image"
            elif isinstance(visuals[0], str):
                task_type = "video"
            run_wall_start = time.perf_counter()
            self._last_sampled_frames = 0
            self._last_video_duration = 0.0
            self._last_effective_fps = 0.0
            self._last_gate_keep_ratio = None
            self._last_recomputed_patches = None
            self._last_orig_patches = None
            self._last_recompute_ratio = None
            self._last_gate_policy = "disabled"
            self._last_gate_metric = "none"
            self._last_gate_projection_mode = self.gate_projection_mode if self.use_gated_tok else "disabled"
            self._last_gate_refresh_interval_frames = self.gate_refresh_interval_frames if self.use_gated_tok else 0
            self._last_forced_refresh_frames = 0
            self._last_forced_refresh_patches = 0
            self._last_forced_refresh_frame_indices = "none"
            self._last_prompt_router = self.prompt_router
            self._last_question_route = "unrouted"
            self._last_video_decode_backend = self.video_decode_backend
            self._last_effective_max_new_tokens = 1024
            self._last_frozen_cache_hit = False
            self._last_frozen_cache_array_sha256 = "none"
            self._last_frozen_cache_video_key = "none"
            self._last_requested_frames = self.max_frames_num
            self._last_reference_frames = self.max_frames_num
            self._last_reference_orig_patches = None
            self._last_patch_projection_recompute_ratio = None
            self._last_patch_projection_compute_ratio_vs_reference = None
            if self._gated_vision_embeddings is not None:
                self._gated_vision_embeddings.set_video_context(None)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            # Copy before consuming ``until`` or adding defaults: Instance
            # request objects can be reused by caches and must remain immutable.
            gen_kwargs = dict(all_gen_kwargs[0])

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            default_max_new_tokens = normalize_positive_integer(gen_kwargs["max_new_tokens"], name="max_new_tokens")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            if task_type == "video":
                request_plan = resolve_video_request_plan(
                    context,
                    prompt_router=self.prompt_router,
                    max_frames_num=self.max_frames_num,
                    ocr_max_frames_num=self.ocr_max_frames_num,
                    gate_policy=self.gate_policy,
                    video_decode_backend=self.video_decode_backend,
                    default_max_new_tokens=default_max_new_tokens,
                    subtitle_video_decode_backend=self.subtitle_video_decode_backend,
                    subtitle_max_new_tokens=self.subtitle_max_new_tokens,
                )
                self._last_prompt_router = request_plan.prompt_router
                self._last_question_route = request_plan.question_route
                self._last_requested_frames = request_plan.requested_frames
                self._last_reference_frames = request_plan.reference_frames
                self._last_video_decode_backend = request_plan.video_decode_backend
                self._last_effective_max_new_tokens = request_plan.max_new_tokens
            else:
                request_plan = VideoRequestPlan(
                    prompt_router=self.prompt_router,
                    question_route=task_type,
                    requested_frames=self.max_frames_num,
                    reference_frames=self.max_frames_num,
                    gate_policy="all",
                    video_decode_backend="not_applicable",
                    max_new_tokens=default_max_new_tokens,
                )

            # Apply the checkpoint's multimodal chat template.  Custom Qwen-VL
            # processors (LLaVA-OneVision 1.5/2) only turn structured
            # ``type=video`` content into the checkpoint's actual video token;
            # a literal "<video>" string tokenizes as plain text and yields an
            # empty prediction after the model rejects the feature/token count.
            messages = self._processor_messages(task_type, context)
            if self.chat_template is not None:
                self.tokenizer.chat_template = self.chat_template
                text = self._image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif getattr(self._image_processor, "apply_chat_template", None) is not None:
                text = self._image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif self.tokenizer.chat_template is not None:
                # Standard HF LLaVA processors accept literal placeholders.
                placeholder = DEFAULT_VIDEO_TOKEN if task_type == "video" else DEFAULT_IMAGE_TOKEN
                text_messages = [{"role": "user", "content": f"{placeholder}\n{context}"}]
                text = self.tokenizer.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                placeholder = DEFAULT_VIDEO_TOKEN if task_type == "video" else DEFAULT_IMAGE_TOKEN
                text_messages = [{"role": "user", "content": f"{placeholder}\n{context}"}]
                text = self.tokenizer.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            processor_owns_video_decode = task_type == "video" and self._processor_owns_video_decode()
            if task_type == "video" and not processor_owns_video_decode:
                try:
                    visuals = [
                        self.load_video(
                            visuals,
                            request_plan.requested_frames,
                            video_decode_backend=request_plan.video_decode_backend,
                        )
                    ]
                except Exception as exc:
                    raise RuntimeError(f"Could not load video {visuals!r}") from exc
            elif task_type == "video" and request_plan.video_decode_backend != self.video_decode_backend:
                raise RuntimeError("Route-specific video_decode_backend is unsupported when the checkpoint processor owns video decoding")

            inputs = None
            if task_type == "image":
                inputs = self._image_processor(images=visuals, text=text, return_tensors="pt").to(self._input_device(), self.model.dtype)
            elif task_type == "video":
                processor_kwargs = {"videos": visuals, "text": [text], "return_tensors": "pt", "padding": True}
                if processor_owns_video_decode:
                    processor_kwargs["num_frames"] = request_plan.requested_frames
                inputs = self._image_processor(**processor_kwargs).to(self._input_device(), self.model.dtype)
                if "second_per_grid_ts" in inputs and not self._model_accepts_kwarg("second_per_grid_ts"):
                    inputs.pop("second_per_grid_ts", None)

            self._set_gated_video_context(task_type, inputs, gate_policy=request_plan.gate_policy)
            if not self.use_gated_tok:
                self._record_ungated_patch_metrics(task_type, inputs)

            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            generation_kwargs = {
                "do_sample": gen_kwargs["temperature"] > 0,
                "temperature": gen_kwargs["temperature"],
                "top_p": gen_kwargs["top_p"],
                "num_beams": gen_kwargs["num_beams"],
                "max_new_tokens": request_plan.max_new_tokens,
                "use_cache": self.use_cache,
                "pad_token_id": self.eot_token_id,
                "eos_token_id": self.eot_token_id,
            }

            def generate_text(options):
                generated = self.model.generate(**inputs, **options)
                generated = generated[:, inputs["input_ids"].shape[-1] :]
                return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

            text_outputs = generate_text(generation_kwargs)
            if not text_outputs:
                retry_kwargs = dict(generation_kwargs)
                retry_kwargs["min_new_tokens"] = min(4, int(request_plan.max_new_tokens))
                eval_logger.warning(f"Decoded response was blank for task={task} doc_id={doc_id[0]}; retrying once")
                text_outputs = generate_text(retry_kwargs)
            if not text_outputs:
                raise RuntimeError(f"Model returned an empty generation for task={task} doc_id={doc_id[0]}")
            if task_type == "video" and self.profiling:
                wall_time_s = time.perf_counter() - run_wall_start
                sampled_frames = int(self._last_sampled_frames)
                throughput_fps = sampled_frames / wall_time_s if wall_time_s > 0 else 0.0
                if self.use_gated_tok:
                    gate_keep_ratio = self._last_gate_keep_ratio if self._last_gate_keep_ratio is not None else 1.0
                    recomputed_patches = self._last_recomputed_patches if self._last_recomputed_patches is not None else 0
                    orig_patches = self._last_orig_patches if self._last_orig_patches is not None else 0
                    recompute_ratio = self._last_recompute_ratio if self._last_recompute_ratio is not None else 1.0
                    gate_policy = self._last_gate_policy or self.gate_policy
                    gate_metric = self._last_gate_metric or self.gate_metric
                    gate_metrics = f"gate_keep_ratio={gate_keep_ratio:.6f} recomputed_patches={recomputed_patches} orig_patches={orig_patches} recompute_ratio={recompute_ratio:.6f} gate_policy={gate_policy} gate_metric={gate_metric}"
                else:
                    gate_keep_ratio = 1.0
                    recomputed_patches = self._last_recomputed_patches if self._last_recomputed_patches is not None else 0
                    orig_patches = self._last_orig_patches if self._last_orig_patches is not None else 0
                    recompute_ratio = 1.0
                    gate_metrics = f"gate_keep_ratio={gate_keep_ratio:.6f} recomputed_patches={recomputed_patches} orig_patches={orig_patches} recompute_ratio={recompute_ratio:.6f} gate_policy=disabled gate_metric=none"
                reference_orig_patches = self._last_reference_orig_patches if self._last_reference_orig_patches is not None else 0
                patch_projection_recompute_ratio = self._last_patch_projection_recompute_ratio if self._last_patch_projection_recompute_ratio is not None else recompute_ratio
                patch_projection_compute_ratio_vs_reference = self._last_patch_projection_compute_ratio_vs_reference if self._last_patch_projection_compute_ratio_vs_reference is not None else 1.0
                gate_projection_mode = self._last_gate_projection_mode or "disabled"
                gate_refresh_interval_frames = int(self._last_gate_refresh_interval_frames or 0)
                forced_refresh_frames = int(self._last_forced_refresh_frames or 0)
                forced_refresh_patches = int(self._last_forced_refresh_patches or 0)
                forced_refresh_frame_indices = self._last_forced_refresh_frame_indices or "none"
                print(
                    "[DENSE_METRICS] "
                    f"sampled_frames={sampled_frames} "
                    f"requested_frames={self._last_requested_frames} "
                    f"reference_frames={self._last_reference_frames} "
                    f"prompt_router={self._last_prompt_router} "
                    f"question_route={self._last_question_route} "
                    f"video_decode_backend={self._last_video_decode_backend} "
                    f"effective_max_new_tokens={self._last_effective_max_new_tokens} "
                    f"frozen_cache_hit={str(self._last_frozen_cache_hit).lower()} "
                    f"frozen_cache_manifest_sha256={self._last_frozen_cache_manifest_sha256} "
                    f"frozen_cache_array_sha256={self._last_frozen_cache_array_sha256} "
                    f"frozen_cache_video_key={self._last_frozen_cache_video_key} "
                    f"effective_fps={self._last_effective_fps:.6f} "
                    f"wall_time_s={wall_time_s:.6f} "
                    f"throughput_fps={throughput_fps:.6f} "
                    "retention_ratio=1.000000 "
                    "pruning_enabled=false prune_mode=off prune_apply_mode=pack "
                    "prune_keep_ratio_actual=1.000000 "
                    "scene_merge_applied=false scene_merge_keep_ratio_actual=1.000000 "
                    f"{gate_metrics} "
                    f"gate_projection_mode={gate_projection_mode} "
                    "gate_refresh_interval_frames="
                    f"{gate_refresh_interval_frames} "
                    f"forced_refresh_frames={forced_refresh_frames} "
                    f"forced_refresh_patches={forced_refresh_patches} "
                    "forced_refresh_frame_indices="
                    f"{forced_refresh_frame_indices} "
                    f"reference_orig_patches={reference_orig_patches} "
                    f"patch_projection_recompute_ratio={patch_projection_recompute_ratio:.6f} "
                    "patch_projection_compute_ratio_vs_reference="
                    f"{patch_projection_compute_ratio_vs_reference:.6f} "
                    "merge_ratio=1.000000",
                    flush=True,
                )
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")

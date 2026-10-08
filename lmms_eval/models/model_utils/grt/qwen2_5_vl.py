import base64
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import av
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.grt.load_video import (
    _stream_total_frames,
    read_video_pyav_seek_base64,
)

decord, _has_decord = optional_import("decord")
process_vision_info, _has_qwen_vl_utils = optional_import("qwen_vl_utils", "process_vision_info")


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_optional_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return float(value)


def _read_video_metadata_pyav(video_path: str) -> Tuple[int, float, float]:
    """Return frame count, average FPS, and duration using the sampling backend."""
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        orig_fps = float(stream.average_rate) if stream.average_rate else 0.0
        if orig_fps <= 0:
            raise ValueError(f"Cannot determine video FPS: {video_path}")
        total_frames = _stream_total_frames(stream, orig_fps)

    duration_sec = total_frames / orig_fps
    return total_frames, orig_fps, duration_sec


def _count_feature_tokens(outputs) -> Optional[int]:
    """Count visual tokens returned by Qwen's video feature extractor."""
    pooler_output = getattr(outputs, "pooler_output", None)
    features = pooler_output if pooler_output is not None else outputs

    if isinstance(features, (list, tuple)):
        counts = []
        for feature in features:
            shape = getattr(feature, "shape", None)
            if shape is not None and len(shape) > 0:
                counts.append(int(shape[0]))
        return sum(counts) if counts else None

    shape = getattr(features, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return None


def _count_grid_tokens(grid_thw) -> Optional[int]:
    """Count pre-merge Conv3d patch tubes described by Qwen's THW grid."""
    if grid_thw is None:
        return None
    rows = grid_thw.detach().cpu().tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
    try:
        return sum(int(t) * int(h) * int(w) for t, h, w in rows)
    except (TypeError, ValueError):
        return None


class GatedQwenVisionPatchEmbed(nn.Module):
    """
    Drop-in wrapper for Qwen2.5-VL's Conv3d patch embedding.

    It keeps Qwen's input and output sequence shapes unchanged, but avoids
    recomputing visual patch embeddings for later temporal patch groups whose
    raw patch-tube difference is below the motion threshold.
    """

    def __init__(
        self,
        orig_patch_embed: nn.Module,
        diff_threshold: float,
        gate_policy: str = "motion",
        random_keep_ratio: Optional[float] = None,
        random_seed: int = 0,
        parent=None,
    ) -> None:
        super().__init__()
        self.proj = orig_patch_embed.proj
        self.patch_size = orig_patch_embed.patch_size
        self.temporal_patch_size = orig_patch_embed.temporal_patch_size
        self.in_channels = orig_patch_embed.in_channels
        self.embed_dim = orig_patch_embed.embed_dim
        self.diff_threshold = float(diff_threshold)
        self.gate_policy = str(gate_policy or "motion").lower()
        if self.gate_policy not in {"motion", "random", "all"}:
            raise ValueError(f"Unsupported gate_policy={gate_policy}. Expected motion|random|all.")
        self.random_keep_ratio = 0.3 if random_keep_ratio is None else float(random_keep_ratio)
        self.random_keep_ratio = min(max(self.random_keep_ratio, 0.0), 1.0)
        self.random_seed = int(random_seed)
        self.parent = parent
        self._random_call_idx = 0
        self._current_grid_thw = None
        self._current_is_video = False
        self.last_keep_flat = None
        self.last_diff_score_flat = None

    def set_grid_thw(self, grid_thw, *, is_video: bool) -> None:
        self._current_grid_thw = grid_thw
        self._current_is_video = bool(is_video)

    def clear_grid_thw(self) -> None:
        self._current_grid_thw = None
        self._current_is_video = False

    def _grid_rows(self):
        if self._current_grid_thw is None:
            return None
        if isinstance(self._current_grid_thw, torch.Tensor):
            return self._current_grid_thw.detach().cpu().tolist()
        return self._current_grid_thw

    def _record_metrics(self, recomputed_patches: int, orig_patches: int) -> None:
        ratio = recomputed_patches / max(orig_patches, 1)
        if self.parent is not None:
            reference_orig_patches = int(getattr(self.parent, "_last_reference_orig_patches", orig_patches) or orig_patches)
            self.parent._last_gate_keep_ratio = float(ratio)
            self.parent._last_recomputed_patches = int(recomputed_patches)
            self.parent._last_orig_patches = int(orig_patches)
            self.parent._last_recompute_ratio = float(ratio)
            self.parent._last_gate_policy = self.gate_policy
            self.parent._last_gate_metric = "l2"
            self.parent._last_reference_orig_patches = reference_orig_patches
            self.parent._last_patch_projection_recompute_ratio = float(ratio)
            self.parent._last_patch_projection_compute_ratio_vs_reference = recomputed_patches / max(reference_orig_patches, 1)
            if getattr(self.parent, "profiling", False):
                print(
                    f"[QWEN_GRT_METRICS] gate_policy={self.gate_policy} recomputed_patches={int(recomputed_patches)} orig_patches={int(orig_patches)} recompute_ratio={ratio:.6f}",
                    flush=True,
                )

    def _full_forward(self, patches_5d: torch.Tensor, *, record_video: bool) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        out = self.proj(patches_5d.to(dtype=target_dtype)).view(-1, self.embed_dim)
        if record_video:
            self.last_keep_flat = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
            self.last_diff_score_flat = torch.ones(out.shape[0], dtype=torch.float32, device=out.device)
            self._record_metrics(out.shape[0], out.shape[0])
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        patches_5d = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        rows = self._grid_rows()
        token_count = int(patches_5d.shape[0])
        if not self._current_is_video or rows is None:
            return self._full_forward(patches_5d, record_video=False)

        expected_tokens = sum(int(t) * int(h) * int(w) for t, h, w in rows)
        if expected_tokens != token_count:
            return self._full_forward(patches_5d, record_video=True)
        if self.gate_policy == "all" or self.diff_threshold < 0:
            # Use the original Conv3d exactly.  This makes the all-patches row a
            # real numerical control instead of an F.linear approximation.
            return self._full_forward(patches_5d, record_video=True)

        patches_flat = patches_5d.to(dtype=target_dtype).reshape(token_count, -1)
        weight_flat = self.proj.weight.view(self.embed_dim, -1)
        bias = self.proj.bias
        outputs = []
        keep_parts = []
        score_parts = []
        offset = 0
        recomputed_patches = 0
        orig_patches = 0

        for grid_t, grid_h, grid_w in rows:
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            spatial_tokens = grid_h * grid_w
            segment_tokens = grid_t * spatial_tokens
            segment = patches_flat[offset : offset + segment_tokens]
            offset += segment_tokens
            orig_patches += segment_tokens
            if segment_tokens == 0:
                continue

            segment = segment.view(grid_t, spatial_tokens, -1)
            keep = torch.zeros((grid_t, spatial_tokens), dtype=torch.bool, device=segment.device)
            scores = torch.zeros((grid_t, spatial_tokens), dtype=torch.float32, device=segment.device)
            keep[0] = True
            scores[0] = 1.0

            if grid_t > 1:
                if self.gate_policy == "random":
                    generator_device = segment.device if segment.device.type == "cuda" else "cpu"
                    generator = torch.Generator(device=generator_device)
                    generator.manual_seed(self.random_seed + self._random_call_idx)
                    self._random_call_idx += 1
                    rand = torch.rand((grid_t - 1, spatial_tokens), device=segment.device, generator=generator)
                    keep[1:] = rand < self.random_keep_ratio
                    scores[1:] = rand.float()
                else:
                    # Compare against the raw patch tube that produced the
                    # cached embedding. Adjacent-frame comparisons allow many
                    # small changes to accumulate while reusing a stale value.
                    cached_source = segment[0].clone()
                    for t_idx in range(1, grid_t):
                        diffs = (segment[t_idx].float() - cached_source.float()).norm(dim=-1)
                        selected = diffs > self.diff_threshold
                        keep[t_idx] = selected
                        scores[t_idx] = diffs / diffs.amax().clamp_min(1e-6)
                        cached_source = torch.where(
                            selected.unsqueeze(-1),
                            segment[t_idx],
                            cached_source,
                        )

            out_segment = torch.empty((grid_t, spatial_tokens, self.embed_dim), device=segment.device, dtype=target_dtype)
            segment_start = offset - segment_tokens
            first_tubes = patches_5d[segment_start : segment_start + spatial_tokens]
            out_segment[0] = self.proj(first_tubes.to(dtype=target_dtype)).view(-1, self.embed_dim)
            recomputed_patches += spatial_tokens

            for t_idx in range(1, grid_t):
                out_segment[t_idx] = out_segment[t_idx - 1]
                selected = keep[t_idx]
                selected_count = int(selected.sum().item())
                if selected.all():
                    tube_start = segment_start + t_idx * spatial_tokens
                    tubes = patches_5d[tube_start : tube_start + spatial_tokens]
                    out_segment[t_idx] = self.proj(tubes.to(dtype=target_dtype)).view(-1, self.embed_dim)
                    recomputed_patches += spatial_tokens
                elif selected_count:
                    out_segment[t_idx, selected] = F.linear(segment[t_idx, selected], weight_flat, bias)
                    recomputed_patches += selected_count

            outputs.append(out_segment.reshape(segment_tokens, self.embed_dim))
            keep_parts.append(keep.reshape(-1))
            score_parts.append(scores.reshape(-1))

        if not outputs:
            return self._full_forward(patches_5d, record_video=True)

        self.last_keep_flat = torch.cat(keep_parts, dim=0)
        self.last_diff_score_flat = torch.cat(score_parts, dim=0)
        self._record_metrics(recomputed_patches, orig_patches)
        return torch.cat(outputs, dim=0)


class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        revision: str = "main",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        use_gated_tok: Optional[bool] = False,
        gate_diff_threshold: Optional[float] = 0.3,
        gate_policy: Optional[str] = "motion",
        random_keep_ratio: Optional[float] = None,
        random_seed: Optional[int] = 0,
        profiling: Optional[bool] = False,
        grt_enable: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Legacy no-op args from the initial qwen2_5_vl_grt frame-selection draft.
        kwargs.pop("grt_candidate_frames", None)
        kwargs.pop("grt_downsample_size", None)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.revision = str(revision)
        self.use_custom_video_loader = _as_bool(use_custom_video_loader)
        self.fps = fps
        if isinstance(self.fps, str) and self.fps.strip():
            self.fps = float(self.fps)
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = None if max_image_size is None else int(max_image_size)
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")
        self.use_gated_tok = _as_bool(use_gated_tok) or (_as_bool(grt_enable) if grt_enable is not None else False)
        self.gate_diff_threshold = float(_as_optional_float(gate_diff_threshold, 0.3))
        self.gate_policy = str(gate_policy or "motion").lower()
        self.random_keep_ratio = _as_optional_float(random_keep_ratio, None)
        self.random_seed = int(random_seed)
        self.profiling = _as_bool(profiling)
        self._last_pre_tokens = 0
        self._last_post_tokens = 0
        self._last_tokenization_time = 0.0
        self._last_gate_keep_ratio = 1.0
        self._last_recomputed_patches = 0
        self._last_orig_patches = 0
        self._last_recompute_ratio = 1.0
        self._last_gate_policy = self.gate_policy if self.use_gated_tok else "disabled"
        self._last_gate_metric = "l2" if self.use_gated_tok else "none"
        self._last_sampled_frames = 0
        self._last_requested_frames = self.max_num_frames if hasattr(self, "max_num_frames") else int(max_num_frames)
        self._last_reference_frames = self._last_requested_frames
        self._last_effective_fps = 0.0
        self._last_reference_orig_patches = 0
        self._last_patch_projection_recompute_ratio = 1.0
        self._last_patch_projection_compute_ratio_vs_reference = 1.0

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype="auto",
                device_map=self.device_map,
            ).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = int(max_num_frames)
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            revision=revision,
            padding_side="left",
        )
        # Install the non-mutating video feature counter for both base and GRT.
        # Only the GRT branch replaces the patch embedding itself.
        self._install_qwen_video_hooks()

        self._config = self.model.config
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

    def _install_qwen_video_hooks(self) -> None:
        core_model = self._model.model
        if getattr(core_model, "_densevideo_qwen_video_hooks", False):
            return

        old_patch_embed = core_model.visual.patch_embed
        if self.use_gated_tok and not isinstance(old_patch_embed, GatedQwenVisionPatchEmbed):
            core_model.visual.patch_embed = GatedQwenVisionPatchEmbed(
                old_patch_embed,
                diff_threshold=self.gate_diff_threshold,
                gate_policy=self.gate_policy,
                random_keep_ratio=self.random_keep_ratio,
                random_seed=self.random_seed,
                parent=self,
            )

        original_get_video_features = core_model.get_video_features

        def patched_get_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
            patch_embed = core_model.visual.patch_embed
            orig_patches = _count_grid_tokens(video_grid_thw)
            if orig_patches is not None:
                self._last_pre_tokens = int(orig_patches)
                self._last_orig_patches = int(orig_patches)
                self._last_reference_orig_patches = int(orig_patches)
                if not self.use_gated_tok:
                    self._last_recomputed_patches = int(orig_patches)
                    self._last_recompute_ratio = 1.0
                    self._last_patch_projection_recompute_ratio = 1.0
                    self._last_patch_projection_compute_ratio_vs_reference = 1.0
            if hasattr(patch_embed, "set_grid_thw"):
                patch_embed.set_grid_thw(video_grid_thw, is_video=True)
            start = time.perf_counter()
            try:
                outputs = original_get_video_features(
                    pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    **kwargs,
                )
                self._last_tokenization_time = float(time.perf_counter() - start)
                self._last_pre_tokens = int(self._last_orig_patches or orig_patches or 0)
                post_tokens = _count_feature_tokens(outputs)
                if post_tokens is not None:
                    self._last_post_tokens = post_tokens
                return outputs
            finally:
                if hasattr(patch_embed, "clear_grid_thw"):
                    patch_embed.clear_grid_thw()

        core_model.get_video_features = patched_get_video_features
        core_model._densevideo_qwen_video_hooks = True

    def _reset_dense_metrics(self) -> None:
        self._last_pre_tokens = 0
        self._last_post_tokens = 0
        self._last_tokenization_time = 0.0
        self._last_gate_keep_ratio = 1.0
        self._last_recomputed_patches = 0
        self._last_orig_patches = 0
        self._last_recompute_ratio = 1.0
        self._last_gate_policy = self.gate_policy if self.use_gated_tok else "disabled"
        self._last_gate_metric = "l2" if self.use_gated_tok else "none"
        self._last_sampled_frames = 0
        self._last_requested_frames = self.max_num_frames
        self._last_reference_frames = self.max_num_frames
        self._last_effective_fps = 0.0
        self._last_reference_orig_patches = 0
        self._last_patch_projection_recompute_ratio = 1.0
        self._last_patch_projection_compute_ratio_vs_reference = 1.0

    def _print_dense_metrics(self, wall_time_s: float) -> None:
        if not self.profiling:
            return
        sampled_frames = int(self._last_sampled_frames or 0)
        pre_tokens = int(self._last_pre_tokens or self._last_orig_patches or 0)
        post_tokens = int(self._last_post_tokens or 0)
        recomputed_patches = int(self._last_recomputed_patches or pre_tokens)
        orig_patches = int(self._last_orig_patches or pre_tokens)
        retention_ratio = post_tokens / max(pre_tokens, 1)
        throughput_fps = sampled_frames / max(wall_time_s, 1e-6)
        print(
            "[DENSE_METRICS] "
            f"sampled_frames={sampled_frames} "
            f"requested_frames={int(self._last_requested_frames)} "
            f"reference_frames={int(self._last_reference_frames)} "
            f"effective_fps={float(self._last_effective_fps):.6f} "
            f"wall_time_s={float(wall_time_s):.6f} "
            f"throughput_fps={throughput_fps:.6f} "
            f"pre_tokens={pre_tokens} "
            f"post_tokens={post_tokens} "
            f"retention_ratio={retention_ratio:.6f} "
            "pruning_enabled=false "
            "prune_mode=off "
            "prune_apply_mode=pack "
            f"post_tokens_before_prune={post_tokens} "
            f"post_tokens_after_prune={post_tokens} "
            "prune_keep_ratio_actual=1.000000 "
            f"gate_keep_ratio={float(self._last_gate_keep_ratio):.6f} "
            f"recomputed_patches={recomputed_patches} "
            f"orig_patches={orig_patches} "
            f"recompute_ratio={float(self._last_recompute_ratio):.6f} "
            f"gate_policy={self._last_gate_policy} "
            f"gate_metric={self._last_gate_metric} "
            f"reference_orig_patches={int(self._last_reference_orig_patches)} "
            "patch_projection_recompute_ratio="
            f"{float(self._last_patch_projection_recompute_ratio):.6f} "
            "patch_projection_compute_ratio_vs_reference="
            f"{float(self._last_patch_projection_compute_ratio_vs_reference):.6f} "
            "merge_ratio=1.000000 "
            f"tokenization_time_s={float(self._last_tokenization_time):.6f}",
            flush=True,
        )

    def _load_video_base64(self, visual: str):
        total_frames, orig_fps, duration_sec = _read_video_metadata_pyav(visual)
        frames = read_video_pyav_seek_base64(
            visual,
            num_frm=self.max_num_frames,
            fps=self.fps,
            img_format="JPEG",
            max_image_size=self.max_image_size,
        )
        self._last_sampled_frames = len(frames)
        self._last_requested_frames = self.max_num_frames
        self._last_reference_frames = min(self.max_num_frames, total_frames)
        self._last_effective_fps = len(frames) / duration_sec if duration_sec > 0 else 0.0
        if self.profiling:
            print(
                "[FPS_STATS] strategy=uniform "
                f"target_fps={self.fps if self.fps is not None else 'none'} "
                f"orig_fps={orig_fps:.6f} duration_s={duration_sec:.6f} "
                f"total_frames={total_frames} capped_frames={total_frames} "
                f"sampled_frames={len(frames)} effective_fps={self._last_effective_fps:.6f}",
                flush=True,
            )
        return frames

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
            # End-to-end request timing begins before document visual loading,
            # video decoding, processor work, host-to-device transfer, and
            # generation.  Keep effective_fps separate: it remains the sampled
            # frame density over source-video duration.
            run_wall_start = time.perf_counter()
            self._reset_dense_metrics()
            chunk_has_video = False
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # if isinstance(contexts, tuple):
            #     contexts = list(contexts)

            # for i in range(len(contexts)):
            #     for j in range(32):
            #         if f"<image {j}>" in contexts[i]:
            #             contexts[i] = contexts[i].replace(f"<image {j}>", "<image>")
            #         if f"\\<image {j}\\>" in contexts[i]:
            #             contexts[i] = contexts[i].replace(f"\\<image {j}\\>", "<image>")
            # if "<image>" in contexts[i]:
            #     contexts[i] = contexts[i].replace("<image>", "")
            # print(contexts[i])

            # for i in range(len(contexts)):
            #     if "<image>" in contexts[i]:
            #         contexts[i] = contexts[i].replace("<image>", "")

            messages = []
            for i, context in enumerate(contexts):
                # context += "\nPlease think step by step."
                # if "<image>" in context:
                #     context = context.replace("<image>", "")

                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        chunk_has_video = True
                        if self.use_custom_video_loader:
                            visual = self._load_video_base64(visual)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                        else:
                            if decord is None:
                                raise ImportError("Install decord or set use_custom_video_loader=True for the PyAV path.")
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": 360 * 420}, {"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)
            # print("message")

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                # fps=self.fps,
                padding=True,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans
            wall_time_s = time.perf_counter() - run_wall_start
            if chunk_has_video:
                self._print_dense_metrics(wall_time_s)

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


class Qwen2_5_VL_GRT(Qwen2_5_VL):
    """Qwen2.5-VL baseline with GRT-style patch embedding recompute reuse."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("use_custom_video_loader", True)
        kwargs.setdefault("max_num_frames", 64)
        kwargs.setdefault("use_gated_tok", True)
        kwargs.setdefault("gate_policy", "motion")
        kwargs.setdefault("gate_diff_threshold", 0.3)
        kwargs.setdefault("profiling", True)
        super().__init__(*args, **kwargs)

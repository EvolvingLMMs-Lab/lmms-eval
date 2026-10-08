"""Prompt-routed GRT thresholds for the DIVE Qwen2.5-VL study.

This isolated module leaves the upstream Qwen model unchanged.
Only the public first sentence of each request selects a threshold.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from lmms_eval.models.model_utils.grt.qwen2_5_vl import (
    GatedQwenVisionPatchEmbed,
    Qwen2_5_VL,
)

DENSEVIDEO_LPM_PROMPT_ROUTER = "densevideo_lpm_first_sentence_v1"

_SUBTITLE_QUESTION_PREFIX = re.compile(
    r"^what\s+subtitles\s+appear\s+in\s+the\s+entire\s+video(?:\s|$)",
    flags=re.IGNORECASE,
)
_OCR_QUESTION_PREFIX = re.compile(
    r"^what\s+text\s+is\s+extracted\s+by\s+ocr\s+in\s+the\s+entire\s+video(?:\s|$)",
    flags=re.IGNORECASE,
)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _finite_unit_interval(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def classify_densevideo_lpm_first_sentence(context: object) -> str:
    """Return ``subtitle``, ``ocr``, or fail-closed ``unknown``.

    The classifier intentionally cannot see task names, document ids, qids,
    references, or labels.  Normalizing whitespace is the only preprocessing.
    """

    first_sentence = re.split(r"[.!?]", str(context or ""), maxsplit=1)[0]
    first_sentence = " ".join(first_sentence.strip().split())
    if _SUBTITLE_QUESTION_PREFIX.match(first_sentence):
        return "subtitle"
    if _OCR_QUESTION_PREFIX.match(first_sentence):
        return "ocr"
    return "unknown"


class DualRouteGatedQwenVisionPatchEmbed(GatedQwenVisionPatchEmbed):
    """A request-local policy overlay over the frozen Qwen GRT patch embed."""

    def __init__(self, *args, **kwargs) -> None:
        self._active_gate_policy: Optional[str] = None
        self._active_diff_threshold: Optional[float] = None
        super().__init__(*args, **kwargs)

    @property
    def gate_policy(self) -> str:
        active = getattr(self, "_active_gate_policy", None)
        if active is not None:
            return active
        return self._configured_gate_policy

    @gate_policy.setter
    def gate_policy(self, value: object) -> None:
        policy = str(value or "all").strip().lower()
        if policy not in {"motion", "random", "all"}:
            raise ValueError(f"Unsupported gate_policy={value}. Expected motion|random|all.")
        self._configured_gate_policy = policy

    @property
    def diff_threshold(self) -> float:
        active = getattr(self, "_active_diff_threshold", None)
        if active is not None:
            return active
        return self._configured_diff_threshold

    @diff_threshold.setter
    def diff_threshold(self, value: object) -> None:
        self._configured_diff_threshold = float(value)

    @property
    def active_gate_policy(self) -> Optional[str]:
        return self._active_gate_policy

    @property
    def active_diff_threshold(self) -> Optional[float]:
        return self._active_diff_threshold

    @property
    def configured_gate_policy(self) -> str:
        return self._configured_gate_policy

    @property
    def configured_diff_threshold(self) -> float:
        return self._configured_diff_threshold

    def set_route_context(self, route: str, threshold: Optional[float]) -> None:
        if route in {"subtitle", "ocr"}:
            self._active_gate_policy = "motion"
            self._active_diff_threshold = _finite_nonnegative(
                threshold,
                name="request-local gate_diff_threshold",
            )
            return
        if route != "unknown":
            raise ValueError(f"unsupported request route: {route!r}")
        self._active_gate_policy = "all"
        self._active_diff_threshold = None

    def clear_route_context(self) -> None:
        self._active_gate_policy = None
        self._active_diff_threshold = None


class RouteFloorGatedQwenVisionPatchEmbed(DualRouteGatedQwenVisionPatchEmbed):
    """Request-local route floors evaluated before any patch projection.

    The complete motion mask is built first.  If its planned keep ratio is
    strictly below the active route floor, the request takes one native Conv3d
    over every patch tube.  Otherwise this reproduces the existing dual-route
    Conv3d/F.linear projection path without changing sequence shape.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._active_min_gate_keep_ratio: Optional[float] = None
        self._active_question_route = "unknown"
        self.last_floor_decision: Optional[dict[str, object]] = None
        super().__init__(*args, **kwargs)

    @property
    def active_min_gate_keep_ratio(self) -> Optional[float]:
        return self._active_min_gate_keep_ratio

    def set_floor_route_context(
        self,
        route: str,
        threshold: Optional[float],
        min_gate_keep_ratio: Optional[float],
    ) -> None:
        super().set_route_context(route, threshold)
        self._active_question_route = route
        self._active_min_gate_keep_ratio = (
            None
            if route == "unknown"
            else _finite_unit_interval(
                min_gate_keep_ratio,
                name="request-local min_gate_keep_ratio",
            )
        )

    def clear_route_context(self) -> None:
        super().clear_route_context()
        self._active_min_gate_keep_ratio = None
        self._active_question_route = "unknown"

    def _record_floor_decision(
        self,
        *,
        planned_recomputed_patches: int,
        orig_patches: int,
        triggered: bool,
        actual_recomputed_patches: int,
        actual_gate_policy: str,
        actual_projection: str,
    ) -> None:
        planned_ratio = planned_recomputed_patches / max(orig_patches, 1)
        actual_ratio = actual_recomputed_patches / max(orig_patches, 1)
        floor = self._active_min_gate_keep_ratio
        decision = {
            "question_route": self._active_question_route,
            "min_gate_keep_ratio": floor,
            "planned_gate_keep_ratio": planned_ratio,
            "planned_recomputed_patches": int(planned_recomputed_patches),
            "orig_patches": int(orig_patches),
            "triggered": bool(triggered),
            "actual_recomputed_patches": int(actual_recomputed_patches),
            "actual_gate_keep_ratio": actual_ratio,
            "actual_gate_policy": actual_gate_policy,
            "actual_projection": actual_projection,
        }
        self.last_floor_decision = decision
        parent = self.parent
        if parent is not None:
            parent._last_floor_decision = dict(decision)
        if parent is not None and getattr(parent, "profiling", False):
            floor_label = "none" if floor is None else f"{floor:.6f}"
            print(
                "[DENSE_GATE_FLOOR] "
                f"question_route={self._active_question_route} "
                f"min_gate_keep_ratio={floor_label} "
                f"planned_gate_keep_ratio={planned_ratio:.6f} "
                f"planned_recomputed_patches={int(planned_recomputed_patches)} "
                f"orig_patches={int(orig_patches)} "
                f"triggered={str(bool(triggered)).lower()} "
                f"actual_recomputed_patches={int(actual_recomputed_patches)} "
                f"actual_gate_keep_ratio={actual_ratio:.6f} "
                f"actual_gate_policy={actual_gate_policy} "
                f"actual_projection={actual_projection}",
                flush=True,
            )

    def _native_floor_fallback(
        self,
        patches_5d: torch.Tensor,
        *,
        planned_recomputed_patches: int,
        orig_patches: int,
    ) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # Exactly one original Conv3d invocation, after the full planned mask
        # and its request-level ratio have already been determined.
        out = self.proj(patches_5d.to(dtype=target_dtype)).view(-1, self.embed_dim)
        self.last_keep_flat = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
        self.last_diff_score_flat = torch.ones(out.shape[0], dtype=torch.float32, device=out.device)
        # _record_metrics reads gate_policy.  Temporarily expose the actual
        # fallback policy so both QWEN_GRT_METRICS and DENSE_METRICS say all.
        active_policy = self._active_gate_policy
        self._active_gate_policy = "all"
        try:
            self._record_metrics(orig_patches, orig_patches)
        finally:
            self._active_gate_policy = active_policy
        self._record_floor_decision(
            planned_recomputed_patches=planned_recomputed_patches,
            orig_patches=orig_patches,
            triggered=True,
            actual_recomputed_patches=orig_patches,
            actual_gate_policy="all",
            actual_projection="native_on_full",
        )
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
            return self._full_forward(patches_5d, record_video=True)

        patches_flat = patches_5d.to(dtype=target_dtype).reshape(token_count, -1)
        segments: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = []
        keep_parts = []
        score_parts = []
        offset = 0
        planned_recomputed_patches = 0
        orig_patches = 0

        # Phase 1: build the complete request mask.  No Conv3d or F.linear is
        # called in this phase.
        for grid_t, grid_h, grid_w in rows:
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            spatial_tokens = grid_h * grid_w
            segment_tokens = grid_t * spatial_tokens
            segment_start = offset
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
                    rand = torch.rand(
                        (grid_t - 1, spatial_tokens),
                        device=segment.device,
                        generator=generator,
                    )
                    keep[1:] = rand < self.random_keep_ratio
                    scores[1:] = rand.float()
                else:
                    cached_source = segment[0].clone()
                    for t_idx in range(1, grid_t):
                        diffs = (segment[t_idx].float() - cached_source.float()).norm(dim=-1)
                        selected = diffs > self.diff_threshold
                        keep[t_idx] = selected
                        scores[t_idx] = diffs / diffs.amax().clamp_min(1e-6)
                        cached_source = torch.where(selected.unsqueeze(-1), segment[t_idx], cached_source)
            planned_recomputed_patches += int(keep.sum().item())
            segments.append((segment, keep, scores, segment_start, segment_tokens))
            keep_parts.append(keep.reshape(-1))
            score_parts.append(scores.reshape(-1))

        if not segments or orig_patches <= 0:
            return self._full_forward(patches_5d, record_video=True)
        planned_ratio = planned_recomputed_patches / orig_patches
        floor = self._active_min_gate_keep_ratio
        if floor is not None and planned_ratio < floor:
            return self._native_floor_fallback(
                patches_5d,
                planned_recomputed_patches=planned_recomputed_patches,
                orig_patches=orig_patches,
            )

        # Phase 2: project only after the request-level fallback decision.
        weight_flat = self.proj.weight.view(self.embed_dim, -1)
        bias = self.proj.bias
        outputs = []
        for segment, keep, _scores, segment_start, segment_tokens in segments:
            grid_t, spatial_tokens = keep.shape
            out_segment = torch.empty(
                (grid_t, spatial_tokens, self.embed_dim),
                device=segment.device,
                dtype=target_dtype,
            )
            first_tubes = patches_5d[segment_start : segment_start + spatial_tokens]
            out_segment[0] = self.proj(first_tubes.to(dtype=target_dtype)).view(-1, self.embed_dim)
            for t_idx in range(1, grid_t):
                out_segment[t_idx] = out_segment[t_idx - 1]
                selected = keep[t_idx]
                selected_count = int(selected.sum().item())
                if selected.all():
                    tube_start = segment_start + t_idx * spatial_tokens
                    tubes = patches_5d[tube_start : tube_start + spatial_tokens]
                    out_segment[t_idx] = self.proj(tubes.to(dtype=target_dtype)).view(-1, self.embed_dim)
                elif selected_count:
                    out_segment[t_idx, selected] = F.linear(segment[t_idx, selected], weight_flat, bias)
            outputs.append(out_segment.reshape(segment_tokens, self.embed_dim))

        self.last_keep_flat = torch.cat(keep_parts, dim=0)
        self.last_diff_score_flat = torch.cat(score_parts, dim=0)
        self._record_metrics(planned_recomputed_patches, orig_patches)
        self._record_floor_decision(
            planned_recomputed_patches=planned_recomputed_patches,
            orig_patches=orig_patches,
            triggered=False,
            actual_recomputed_patches=planned_recomputed_patches,
            actual_gate_policy="motion",
            actual_projection="linear_consistent",
        )
        return torch.cat(outputs, dim=0)


class Qwen2_5_VL_DualRoute(Qwen2_5_VL):
    """Qwen2.5-VL with request-local subtitle/OCR GRT thresholds."""

    def __init__(
        self,
        *,
        prompt_router: object = DENSEVIDEO_LPM_PROMPT_ROUTER,
        subtitle_gate_diff_threshold: object,
        ocr_gate_diff_threshold: object,
        **kwargs,
    ) -> None:
        self.prompt_router = str(prompt_router or "").strip().lower()
        self.subtitle_gate_diff_threshold = _finite_nonnegative(
            subtitle_gate_diff_threshold,
            name="subtitle_gate_diff_threshold",
        )
        self.ocr_gate_diff_threshold = _finite_nonnegative(
            ocr_gate_diff_threshold,
            name="ocr_gate_diff_threshold",
        )
        if self.prompt_router != DENSEVIDEO_LPM_PROMPT_ROUTER:
            raise ValueError(f"qwen2_5_vl_dual_route requires prompt_router={DENSEVIDEO_LPM_PROMPT_ROUTER}")
        if not _as_bool(kwargs.get("use_gated_tok", False)):
            raise ValueError("qwen2_5_vl_dual_route requires use_gated_tok=True")
        if str(kwargs.get("gate_policy", "motion") or "motion").strip().lower() != "motion":
            raise ValueError("qwen2_5_vl_dual_route requires gate_policy=motion")
        if not _as_bool(kwargs.get("use_custom_video_loader", False)):
            raise ValueError("qwen2_5_vl_dual_route requires use_custom_video_loader=True")

        # The frozen parent installs its standard wrapper and feature hook.
        # Replace only that wrapper with an additive overlay around the exact
        # same Conv3d projection; the already-installed hook dereferences the
        # current patch_embed for every request.
        super().__init__(**kwargs)
        core_model = self._model.model
        installed = core_model.visual.patch_embed
        if not isinstance(installed, GatedQwenVisionPatchEmbed):
            raise RuntimeError("frozen Qwen wrapper did not install its GRT patch embed")
        dual = DualRouteGatedQwenVisionPatchEmbed(
            installed,
            diff_threshold=self.ocr_gate_diff_threshold,
            # Between requests the adapter must be fail-closed, not motion.
            gate_policy="all",
            random_keep_ratio=installed.random_keep_ratio,
            random_seed=installed.random_seed,
            parent=self,
        )
        core_model.visual.patch_embed = dual
        self._dual_route_patch_embed = dual
        self._last_question_route = "unknown"
        self._last_active_gate_diff_threshold: Optional[float] = None

    @staticmethod
    def _request_context(request: object) -> str:
        args = getattr(request, "args", None)
        if args is None and isinstance(request, (list, tuple)):
            args = request
        if not args:
            return ""
        return str(args[0] or "")

    def _activate_route(self, route: str) -> None:
        if route == "subtitle":
            threshold: Optional[float] = self.subtitle_gate_diff_threshold
        elif route == "ocr":
            threshold = self.ocr_gate_diff_threshold
        else:
            route = "unknown"
            threshold = None
        self._dual_route_patch_embed.set_route_context(route, threshold)
        self._last_question_route = route
        self._last_active_gate_diff_threshold = threshold
        if self.profiling:
            threshold_label = "none" if threshold is None else format(threshold, ".17g")
            print(
                f"[DENSE_ROUTE_GATE] prompt_router={self.prompt_router} question_route={route} gate_policy={self._dual_route_patch_embed.gate_policy} gate_diff_threshold={threshold_label}",
                flush=True,
            )

    def _clear_route(self) -> None:
        self._dual_route_patch_embed.clear_route_context()
        self._last_question_route = "unknown"
        self._last_active_gate_diff_threshold = None

    def generate_until(self, requests: Iterable[object]):
        """Evaluate one request per route context, always clearing in ``finally``."""

        responses = []
        for request in requests:
            route = classify_densevideo_lpm_first_sentence(self._request_context(request))
            self._activate_route(route)
            try:
                result = super().generate_until([request])
                if len(result) != 1:
                    raise RuntimeError("frozen Qwen wrapper returned a non-singleton response for a request-local dual-route call")
                responses.extend(result)
            finally:
                self._clear_route()
        return responses


class Qwen2_5_VL_DualRouteFloor(Qwen2_5_VL_DualRoute):
    """Dual-route Qwen with a request-level pre-projection native floor."""

    def __init__(
        self,
        *,
        subtitle_min_gate_keep_ratio: object,
        ocr_min_gate_keep_ratio: object,
        **kwargs,
    ) -> None:
        self.subtitle_min_gate_keep_ratio = _finite_unit_interval(
            subtitle_min_gate_keep_ratio,
            name="subtitle_min_gate_keep_ratio",
        )
        self.ocr_min_gate_keep_ratio = _finite_unit_interval(
            ocr_min_gate_keep_ratio,
            name="ocr_min_gate_keep_ratio",
        )
        super().__init__(**kwargs)
        installed = self._dual_route_patch_embed
        floor_embed = RouteFloorGatedQwenVisionPatchEmbed(
            installed,
            diff_threshold=self.ocr_gate_diff_threshold,
            gate_policy="all",
            random_keep_ratio=installed.random_keep_ratio,
            random_seed=installed.random_seed,
            parent=self,
        )
        self._model.model.visual.patch_embed = floor_embed
        self._dual_route_patch_embed = floor_embed
        self._last_floor_decision: Optional[dict[str, object]] = None

    def _activate_route(self, route: str) -> None:
        if route == "subtitle":
            threshold: Optional[float] = self.subtitle_gate_diff_threshold
            floor: Optional[float] = self.subtitle_min_gate_keep_ratio
        elif route == "ocr":
            threshold = self.ocr_gate_diff_threshold
            floor = self.ocr_min_gate_keep_ratio
        else:
            route = "unknown"
            threshold = None
            floor = None
        self._dual_route_patch_embed.set_floor_route_context(route, threshold, floor)
        self._last_question_route = route
        self._last_active_gate_diff_threshold = threshold
        if self.profiling:
            threshold_label = "none" if threshold is None else format(threshold, ".17g")
            floor_label = "none" if floor is None else format(floor, ".17g")
            print(
                f"[DENSE_ROUTE_GATE] prompt_router={self.prompt_router} question_route={route} gate_policy={self._dual_route_patch_embed.gate_policy} gate_diff_threshold={threshold_label} min_gate_keep_ratio={floor_label}",
                flush=True,
            )

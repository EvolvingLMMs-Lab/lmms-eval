"""VQToken on LLaVA-OneVision."""

from __future__ import annotations

from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.simple.llava_onevision import (
    Llava_OneVision,
    best_fit_attn_implementation,
)

_SUPPORTED_SELECTION_METHODS = frozenset({"fixed", "elbow", "silhouette"})
_RUNTIME_INSTALL = "llava[runtime] @ git+https://github.com/Hai-chao-Zhang/VQToken.git@a8e3e13e8415b575556dd779e890b77a74ecf52a"


def _validate_cluster_config(selection_method: str, min_clusters: int, max_clusters: int) -> None:
    if selection_method not in _SUPPORTED_SELECTION_METHODS:
        raise ValueError("vqtoken_selection_method must be one of " f"{sorted(_SUPPORTED_SELECTION_METHODS)}, got {selection_method!r}")
    if isinstance(min_clusters, bool) or not isinstance(min_clusters, int) or min_clusters < 1:
        raise ValueError("vqtoken_min_clusters must be a positive integer")
    if isinstance(max_clusters, bool) or not isinstance(max_clusters, int) or max_clusters < 1:
        raise ValueError("vqtoken_max_clusters must be a positive integer")
    if selection_method != "fixed" and max_clusters < min_clusters:
        raise ValueError("adaptive VQToken requires max_clusters to be greater than or equal to min_clusters")


def _require_vqtoken_runtime() -> object:
    runtime, has_runtime = optional_import("VQToken")
    if not has_runtime:
        raise ImportError(f"VQToken requires its public LLaVA runtime. Install with: uv pip install '{_RUNTIME_INSTALL}'")

    capabilities = getattr(runtime, "VQTOKEN_CAPABILITIES", {})
    modes = set(capabilities.get("modes", ()))
    methods = set(capabilities.get("selection_methods", ()))
    if "centroids" not in modes or not _SUPPORTED_SELECTION_METHODS.issubset(methods):
        raise ImportError("The installed VQToken runtime is too old for lmms-eval. " f"Upgrade with: uv pip install --upgrade '{_RUNTIME_INSTALL}'")
    return runtime


@register_model("vqtoken")
class VQToken(Llava_OneVision):
    """Evaluate the public centroid VQToken checkpoint with LLaVA-OneVision."""

    def __init__(
        self,
        pretrained: str = "haichaozhang/VQ-Token-llava-ov-0.5b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = "llava_qwen",
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config: Optional[str] = None,
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        video_decode_backend: str = "decord",
        vqtoken_selection_method: str = "fixed",
        vqtoken_min_clusters: int = 12,
        vqtoken_max_clusters: int = 32,
        use_embedded_vision: Optional[bool] = None,
        **kwargs,
    ) -> None:
        _validate_cluster_config(vqtoken_selection_method, vqtoken_min_clusters, vqtoken_max_clusters)
        if token_strategy != "single":
            raise ValueError("VQToken supports token_strategy=single for video inputs")
        runtime = _require_vqtoken_runtime()

        if use_embedded_vision is None:
            use_embedded_vision = pretrained.rstrip("/") in {
                "haichaozhang/VQ-Token-llava-ov-0.5b",
                "lmms-lab/llava-onevision-qwen2-0.5b-ov",
            }
            detector = getattr(runtime, "has_embedded_vision_weights", None)
            if not use_embedded_vision and callable(detector):
                use_embedded_vision = detector(pretrained)

        self._vqtoken_overwrite_config = {
            "use_vqtoken": True,
            "vqtoken_mode": "centroids",
            "vqtoken_selection_method": vqtoken_selection_method,
            "vqtoken_min_clusters": vqtoken_min_clusters,
            "vqtoken_max_clusters": vqtoken_max_clusters,
            "use_embedded_vision": use_embedded_vision,
        }
        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            customized_config=customized_config,
            max_frames_num=max_frames_num,
            mm_spatial_pool_stride=mm_spatial_pool_stride,
            mm_spatial_pool_mode=mm_spatial_pool_mode,
            token_strategy=token_strategy,
            video_decode_backend=video_decode_backend,
            **kwargs,
        )

    def _get_model_overwrite_config(self) -> dict[str, object]:
        """Return a copy so the parent cannot mutate the adapter's settings."""
        return dict(self._vqtoken_overwrite_config)

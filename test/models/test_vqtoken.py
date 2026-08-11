from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# decord is optional and unavailable on some lmms-eval test platforms. The
# wrapper tests do not decode video, so provide the same minimal import stub
# used by other model tests before importing the LLaVA-OneVision module.
_decord_stub = types.ModuleType("decord")
_decord_stub.__spec__ = ModuleSpec("decord", loader=None)
_decord_stub.VideoReader = MagicMock()
_decord_stub.cpu = MagicMock()
sys.modules.setdefault("decord", _decord_stub)

import lmms_eval.models.simple.llava_onevision as llava_onevision_module  # noqa: E402
from lmms_eval.models import get_model  # noqa: E402
from lmms_eval.models.simple.llava_onevision import Llava_OneVision  # noqa: E402
from lmms_eval.models.simple.vqtoken import (  # noqa: E402
    VQToken,
    _require_vqtoken_runtime,
    _validate_cluster_config,
)


@pytest.mark.parametrize("method", ["fixed", "elbow", "silhouette"])
def test_vqtoken_passes_public_centroid_config(method: str) -> None:
    runtime = SimpleNamespace(
        VQTOKEN_CAPABILITIES={
            "modes": ("centroids",),
            "selection_methods": ("fixed", "elbow", "silhouette"),
        }
    )
    with (
        patch("lmms_eval.models.simple.vqtoken.optional_import", return_value=(runtime, True)),
        patch.object(Llava_OneVision, "__init__", return_value=None) as parent_init,
    ):
        model = VQToken(vqtoken_selection_method=method, vqtoken_min_clusters=12, vqtoken_max_clusters=32)

    kwargs = parent_init.call_args.kwargs
    assert kwargs["pretrained"] == "haichaozhang/VQ-Token-llava-ov-0.5b"
    assert kwargs["model_name"] == "llava_qwen"
    assert model._get_model_overwrite_config() == {
        "use_vqtoken": True,
        "vqtoken_mode": "centroids",
        "vqtoken_selection_method": method,
        "vqtoken_min_clusters": 12,
        "vqtoken_max_clusters": 32,
        "use_embedded_vision": True,
    }


@pytest.mark.parametrize(
    ("method", "min_clusters", "max_clusters"),
    [("attention", 12, 32), ("fixed", 0, 32), ("elbow", 33, 32), ("fixed", True, 32)],
)
def test_vqtoken_rejects_non_public_or_invalid_config(method: str, min_clusters: int, max_clusters: int) -> None:
    with pytest.raises(ValueError):
        _validate_cluster_config(method, min_clusters, max_clusters)


def test_vqtoken_runtime_is_lazy_and_actionable() -> None:
    with patch("lmms_eval.models.simple.vqtoken.optional_import", return_value=(None, False)):
        with pytest.raises(ImportError, match="Hai-chao-Zhang/VQToken"):
            _require_vqtoken_runtime()


def test_vqtoken_rejects_runtime_without_centroid_capability() -> None:
    runtime = SimpleNamespace(VQTOKEN_CAPABILITIES={"modes": (), "selection_methods": ()})
    with patch("lmms_eval.models.simple.vqtoken.optional_import", return_value=(runtime, True)):
        with pytest.raises(ImportError, match="too old"):
            _require_vqtoken_runtime()


def test_fixed_selection_uses_max_clusters_as_k() -> None:
    _validate_cluster_config("fixed", 12, 8)


def test_vqtoken_resolves_through_v2_registry() -> None:
    assert get_model("vqtoken", force_simple=True) is VQToken


def test_vqtoken_rejects_multiple_video_placeholders() -> None:
    with pytest.raises(ValueError, match="token_strategy=single"):
        VQToken(token_strategy="multiple")


def test_vqtoken_defaults_unknown_checkpoint_to_external_vision() -> None:
    runtime = SimpleNamespace(
        VQTOKEN_CAPABILITIES={
            "modes": ("centroids",),
            "selection_methods": ("fixed", "elbow", "silhouette"),
        }
    )
    with (
        patch("lmms_eval.models.simple.vqtoken.optional_import", return_value=(runtime, True)),
        patch.object(Llava_OneVision, "__init__", return_value=None),
    ):
        model = VQToken(pretrained="local/custom-checkpoint")

    assert model._get_model_overwrite_config()["use_embedded_vision"] is False


def test_vqtoken_detects_embedded_vision_in_local_checkpoint() -> None:
    detector = MagicMock(return_value=True)
    runtime = SimpleNamespace(
        VQTOKEN_CAPABILITIES={
            "modes": ("centroids",),
            "selection_methods": ("fixed", "elbow", "silhouette"),
        },
        has_embedded_vision_weights=detector,
    )
    with (
        patch("lmms_eval.models.simple.vqtoken.optional_import", return_value=(runtime, True)),
        patch.object(Llava_OneVision, "__init__", return_value=None),
    ):
        model = VQToken(pretrained="local/checkpoint")

    detector.assert_called_once_with("local/checkpoint")
    assert model._get_model_overwrite_config()["use_embedded_vision"] is True


def test_parent_loader_merges_only_subclass_overrides() -> None:
    runtime = SimpleNamespace(
        VQTOKEN_CAPABILITIES={
            "modes": ("centroids",),
            "selection_methods": ("fixed", "elbow", "silhouette"),
        }
    )
    accelerator = SimpleNamespace(num_processes=1, local_process_index=0)
    fake_model = MagicMock()
    fake_model.config = SimpleNamespace()
    loader_result = (MagicMock(), fake_model, MagicMock(), 4096)

    with (
        patch("lmms_eval.models.simple.vqtoken.optional_import", return_value=(runtime, True)),
        patch.object(llava_onevision_module, "Accelerator", return_value=accelerator),
        patch.object(llava_onevision_module.AutoConfig, "from_pretrained", return_value=SimpleNamespace()),
        patch.object(llava_onevision_module, "load_pretrained_model", create=True, return_value=loader_result) as loader,
    ):
        Llava_OneVision(pretrained="local/base", model_name="llava_qwen")
        base_overrides = loader.call_args.kwargs["overwrite_config"]
        model = VQToken()
        vqtoken_overrides = loader.call_args.kwargs["overwrite_config"]

    assert base_overrides == {
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
    }
    assert vqtoken_overrides == {
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
        "use_vqtoken": True,
        "vqtoken_mode": "centroids",
        "vqtoken_selection_method": "fixed",
        "vqtoken_min_clusters": 12,
        "vqtoken_max_clusters": 32,
        "use_embedded_vision": True,
    }
    vqtoken_overrides["use_vqtoken"] = False
    assert model._get_model_overwrite_config()["use_vqtoken"] is True

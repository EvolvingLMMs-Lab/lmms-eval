from types import SimpleNamespace

import pytest

from lmms_eval.models.model_utils.grt.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.model_utils.grt.qwen_dual_route import (
    DENSEVIDEO_LPM_PROMPT_ROUTER,
    DualRouteGatedQwenVisionPatchEmbed,
    Qwen2_5_VL_DualRoute,
    classify_densevideo_lpm_first_sentence,
)


class FakePatchEmbed:
    gate_policy = "all"
    active_gate_policy = None
    active_diff_threshold = None

    def set_route_context(self, route, threshold):
        self.gate_policy = "motion" if route != "unknown" else "all"
        self.active_gate_policy = self.gate_policy
        self.active_diff_threshold = threshold

    def clear_route_context(self):
        self.gate_policy = "all"
        self.active_gate_policy = None
        self.active_diff_threshold = None


def bare_model():
    model = object.__new__(Qwen2_5_VL_DualRoute)
    model.prompt_router = DENSEVIDEO_LPM_PROMPT_ROUTER
    model.subtitle_gate_diff_threshold = 10.0
    model.ocr_gate_diff_threshold = 0.3
    model._dual_route_patch_embed = FakePatchEmbed()
    model._last_question_route = "unknown"
    model._last_active_gate_diff_threshold = None
    model.profiling = False
    return model


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("What subtitles appear in the entire video? Return all.", "subtitle"),
        ("WHAT   TEXT IS EXTRACTED BY OCR IN THE ENTIRE VIDEO. Return all.", "ocr"),
        ("Summarize the entire video.", "unknown"),
        ("Prefix. What subtitles appear in the entire video?", "unknown"),
    ],
)
def test_public_first_sentence_classifier(context, expected):
    assert classify_densevideo_lpm_first_sentence(context) == expected


def test_request_routes_are_cleared_after_success(monkeypatch):
    model = bare_model()
    observed = []

    def fake_generate(_self, requests):
        observed.append(
            (
                _self._last_question_route,
                _self._dual_route_patch_embed.gate_policy,
                _self._dual_route_patch_embed.active_diff_threshold,
            )
        )
        return ["answer"]

    monkeypatch.setattr(Qwen2_5_VL, "generate_until", fake_generate)
    requests = [
        SimpleNamespace(args=("What subtitles appear in the entire video?",)),
        SimpleNamespace(args=("What text is extracted by OCR in the entire video?",)),
        SimpleNamespace(args=("Unknown prompt.",)),
    ]
    assert model.generate_until(requests) == ["answer", "answer", "answer"]
    assert observed == [
        ("subtitle", "motion", 10.0),
        ("ocr", "motion", 0.3),
        ("unknown", "all", None),
    ]
    assert model._dual_route_patch_embed.active_gate_policy is None
    assert model._dual_route_patch_embed.active_diff_threshold is None
    assert model._dual_route_patch_embed.gate_policy == "all"
    assert model._last_question_route == "unknown"


def test_request_route_is_cleared_after_failure(monkeypatch):
    model = bare_model()

    def fail(_self, requests):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(Qwen2_5_VL, "generate_until", fail)
    request = SimpleNamespace(args=("What text is extracted by OCR in the entire video?",))
    with pytest.raises(RuntimeError, match="fixture failure"):
        model.generate_until([request])
    assert model._dual_route_patch_embed.active_gate_policy is None
    assert model._dual_route_patch_embed.active_diff_threshold is None
    assert model._dual_route_patch_embed.gate_policy == "all"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "bad"])
def test_thresholds_fail_before_heavy_model_loading(value):
    with pytest.raises(ValueError, match="finite non-negative"):
        Qwen2_5_VL_DualRoute(
            subtitle_gate_diff_threshold=value,
            ocr_gate_diff_threshold=0.3,
            use_gated_tok=True,
            gate_policy="motion",
            use_custom_video_loader=True,
        )


def test_adapter_preconditions_fail_before_heavy_model_loading():
    common = {
        "subtitle_gate_diff_threshold": 10.0,
        "ocr_gate_diff_threshold": 0.3,
    }
    with pytest.raises(ValueError, match="prompt_router"):
        Qwen2_5_VL_DualRoute(
            **common,
            prompt_router="off",
            use_gated_tok=True,
            gate_policy="motion",
            use_custom_video_loader=True,
        )
    with pytest.raises(ValueError, match="use_gated_tok"):
        Qwen2_5_VL_DualRoute(
            **common,
            use_gated_tok=False,
            gate_policy="motion",
            use_custom_video_loader=True,
        )
    with pytest.raises(ValueError, match="gate_policy"):
        Qwen2_5_VL_DualRoute(
            **common,
            use_gated_tok=True,
            gate_policy="all",
            use_custom_video_loader=True,
        )
    with pytest.raises(ValueError, match="use_custom_video_loader"):
        Qwen2_5_VL_DualRoute(
            **common,
            use_gated_tok=True,
            gate_policy="motion",
            use_custom_video_loader=False,
        )


def test_dual_overlay_defaults_fail_closed_and_resets():
    original = SimpleNamespace(
        proj=SimpleNamespace(),
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=16,
    )
    embed = DualRouteGatedQwenVisionPatchEmbed(
        original,
        diff_threshold=30.0,
        gate_policy="all",
    )
    assert embed.configured_gate_policy == "all"
    assert embed.gate_policy == "all"
    embed.set_route_context("subtitle", 3.0)
    assert embed.gate_policy == "motion"
    assert embed.diff_threshold == 3.0
    embed.set_route_context("unknown", None)
    assert embed.gate_policy == "all"
    embed.clear_route_context()
    assert embed.gate_policy == "all"
    assert embed.diff_threshold == 30.0

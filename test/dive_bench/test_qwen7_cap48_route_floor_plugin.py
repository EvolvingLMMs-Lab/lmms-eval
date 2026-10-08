from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmms_eval.models.model_utils.grt.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.model_utils.grt.qwen_dual_route import (
    DENSEVIDEO_LPM_PROMPT_ROUTER,
    Qwen2_5_VL_DualRouteFloor,
    RouteFloorGatedQwenVisionPatchEmbed,
)


class CountingConv3d(nn.Conv3d):
    def __init__(self):
        super().__init__(1, 1, kernel_size=(1, 1, 1), bias=True)
        self.calls = []
        with torch.no_grad():
            self.weight.fill_(2.0)
            self.bias.fill_(0.5)

    def forward(self, value):
        self.calls.append(int(value.shape[0]))
        return super().forward(value)


def make_embed(*, values, threshold, floor):
    projection = CountingConv3d()
    original = SimpleNamespace(
        proj=projection,
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        embed_dim=1,
    )
    parent = SimpleNamespace(profiling=False, _last_reference_orig_patches=len(values))
    embed = RouteFloorGatedQwenVisionPatchEmbed(
        original,
        diff_threshold=threshold,
        gate_policy="all",
        parent=parent,
    )
    embed.set_floor_route_context("subtitle", threshold, floor)
    embed.set_grid_thw([[len(values), 1, 1]], is_video=True)
    hidden = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    return embed, parent, projection, hidden


def test_below_floor_uses_one_native_projection_after_complete_plan():
    # keep=[true,false,false,false] gives 1/4, strictly below the 0.80 floor.
    embed, parent, projection, hidden = make_embed(values=[0.0, 0.1, 0.2, 0.3], threshold=1.0, floor=0.80)
    output = embed(hidden)
    native = 2.0 * hidden + 0.5

    assert torch.equal(output, native)
    assert projection.calls == [4]
    assert parent._last_recomputed_patches == 4
    assert parent._last_gate_policy == "all"
    assert embed.last_floor_decision == {
        "question_route": "subtitle",
        "min_gate_keep_ratio": 0.8,
        "planned_gate_keep_ratio": 0.25,
        "planned_recomputed_patches": 1,
        "orig_patches": 4,
        "triggered": True,
        "actual_recomputed_patches": 4,
        "actual_gate_keep_ratio": 1.0,
        "actual_gate_policy": "all",
        "actual_projection": "native_on_full",
    }


def test_equal_to_floor_is_boundary_negative_and_uses_dual_path():
    # cached-source updates yield keep=[true,true,false,true] = exactly 3/4.
    embed, parent, projection, hidden = make_embed(values=[0.0, 2.0, 2.1, 4.1], threshold=1.0, floor=0.75)
    output = embed(hidden)

    assert embed.last_floor_decision["planned_gate_keep_ratio"] == 0.75
    assert embed.last_floor_decision["triggered"] is False
    assert embed.last_floor_decision["actual_projection"] == "linear_consistent"
    assert parent._last_recomputed_patches == 3
    assert parent._last_gate_policy == "motion"
    # With one spatial tube, every selected temporal slice is an all-selected
    # slice and therefore takes the existing exact Conv3d branch.
    assert projection.calls == [1, 1, 1]
    assert output.flatten().tolist() == pytest.approx([0.5, 4.5, 4.5, 8.7])


class FakeFloorPatchEmbed:
    gate_policy = "all"
    active_min_gate_keep_ratio = None

    def set_floor_route_context(self, route, threshold, floor):
        self.gate_policy = "motion" if route != "unknown" else "all"
        self.route = route
        self.threshold = threshold
        self.active_min_gate_keep_ratio = floor

    def clear_route_context(self):
        self.gate_policy = "all"
        self.route = "unknown"
        self.threshold = None
        self.active_min_gate_keep_ratio = None


def bare_floor_model():
    model = object.__new__(Qwen2_5_VL_DualRouteFloor)
    model.prompt_router = DENSEVIDEO_LPM_PROMPT_ROUTER
    model.subtitle_gate_diff_threshold = 3.0
    model.ocr_gate_diff_threshold = 30.0
    model.subtitle_min_gate_keep_ratio = 0.80
    model.ocr_min_gate_keep_ratio = 0.55
    model._dual_route_patch_embed = FakeFloorPatchEmbed()
    model._last_question_route = "unknown"
    model._last_active_gate_diff_threshold = None
    model.profiling = False
    return model


def test_floor_route_is_request_local_and_cleared(monkeypatch):
    model = bare_floor_model()
    observed = []

    def fake_generate(_self, requests):
        observed.append(
            (
                _self._last_question_route,
                _self._dual_route_patch_embed.threshold,
                _self._dual_route_patch_embed.active_min_gate_keep_ratio,
            )
        )
        return ["answer"]

    monkeypatch.setattr(Qwen2_5_VL, "generate_until", fake_generate)
    requests = [
        SimpleNamespace(args=("What subtitles appear in the entire video?",)),
        SimpleNamespace(args=("What text is extracted by OCR in the entire video?",)),
    ]
    assert model.generate_until(requests) == ["answer", "answer"]
    assert observed == [("subtitle", 3.0, 0.80), ("ocr", 30.0, 0.55)]
    assert model._dual_route_patch_embed.gate_policy == "all"
    assert model._dual_route_patch_embed.active_min_gate_keep_ratio is None


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "bad"])
def test_invalid_floor_fails_before_heavy_model_loading(value):
    with pytest.raises(ValueError, match="finite"):
        Qwen2_5_VL_DualRouteFloor(
            subtitle_min_gate_keep_ratio=value,
            ocr_min_gate_keep_ratio=0.55,
            subtitle_gate_diff_threshold=3.0,
            ocr_gate_diff_threshold=30.0,
            use_gated_tok=True,
            gate_policy="motion",
            use_custom_video_loader=True,
        )

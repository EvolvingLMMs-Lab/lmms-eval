"""Deterministic loading, safety and metric contracts for ShareRobot-Bench."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lmms_eval.tasks.share_robot import utils


def sample(trajectory: bool = False) -> dict:
    return {
        "id": 1,
        "image": ["images/a.jpg"] if trajectory else "images/a.jpg",
        "conversations": [{"from": "human", "value": "<image>Task. Please predict up to 10 key trajectory points to complete the task."}, {"from": "gpt", "value": "[(10, 10), (20, 20)]" if trajectory else "[0, 0, 20, 20]"}],
    }


@pytest.fixture
def media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 200)).save(path)
    monkeypatch.setattr(utils, "_asset_path", lambda kind, image: str(path))
    return path


def test_prompts_and_media(media: Path) -> None:
    for trajectory in (False, True):
        doc = sample(trajectory)
        prompt = utils.share_robot_doc_to_text(doc)
        assert "<image>" not in prompt
        assert utils.share_robot_doc_to_visual(doc)[0].size == (100, 200)
        assert utils.share_robot_doc_to_messages(doc)[0]["content"][1]["text"] == prompt
        assert utils.share_robot_doc_to_target(doc) == doc["conversations"][1]["value"]
    assert utils.share_robot_doc_to_text(sample()).endswith("Only return the box in format: [x1, y1, x2, y2] with no other output.")
    assert "[[x1, y1], [x2, y2], ...]" in utils.share_robot_doc_to_text(sample(True))
    doc = sample(True)
    doc["conversations"] += [{"from": "gpt", "value": "ignored"}]
    assert utils.share_robot_doc_to_target(doc) != "ignored"


@pytest.mark.parametrize("box,expected", [("[0,0,20,20]", 1.0), ("[20,20,30,30]", 0.0), ("[0,0,10,20]", 0.5), ("[0,0,15,20]", 0.75), ("[0,0,18,20]", 0.9)])
def test_iou_and_inclusive_thresholds(box: str, expected: float) -> None:
    scores = utils.affordance_process_results(sample(), [box])
    assert scores["average_iou"] == pytest.approx(expected)
    assert scores["average_precision"] == scores["average_iou"]
    assert scores["iou_at_0_5"] == float(expected >= 0.5)
    assert scores["iou_at_0_75"] == float(expected >= 0.75)
    assert scores["iou_at_0_9"] == float(expected >= 0.9)


@pytest.mark.parametrize("answer", ["", "prose", "[1,2]", "[0,0,1e309,2]", '__import__("os").system("false")'])
def test_invalid_boxes_excluded(answer: str) -> None:
    scores = utils.affordance_process_results(sample(), [answer])
    assert scores["average_iou"] is None
    assert scores["valid_items"] == 0
    assert utils.share_robot_valid_mean([None, 0.5]) == 0.5
    assert utils.share_robot_valid_mean([None]) == 0


@pytest.mark.parametrize("answer", ["[(10, 10), (20, 20)]", "[[.1,.05],[.2,.1]]", "answer: [[10,10],[20,20]]", "[10,10] # a\n[20,20] # b"])
def test_trajectory_formats(answer: str, media: Path) -> None:
    scores = utils.trajectory_process_results(sample(True), [answer])
    assert scores["average_discrete_frechet_distance"] == pytest.approx(0)
    assert scores["average_hausdorff_distance"] == pytest.approx(0)
    assert scores["average_root_mean_square_error"] == pytest.approx(0)
    assert scores["parse_valid_rate"] == 1


def test_fallback_order_and_unequal_lengths(media: Path, tmp_path: Path) -> None:
    target = tmp_path / "should-not-exist"
    scores = utils.trajectory_process_results(sample(True), [f'__import__("pathlib").Path({str(target)!r}).touch()'])
    assert not target.exists()
    assert scores["parse_valid_items"] == 0
    assert scores["average_discrete_frechet_distance"] > 0
    p = np.array([[0.0, 0.0], [1.0, 1.0]])
    dfd, hd, rmse = utils._distances(p, p[::-1])
    assert dfd == pytest.approx(2**0.5) and hd == 0 and rmse == pytest.approx(2**0.5)
    assert utils._distances(p, p[:1])[2] == 0


def test_invalid_ground_truth_fails(media: Path) -> None:
    doc = sample(True)
    doc["conversations"][1]["value"] = "bad"
    with pytest.raises(ValueError, match="ground truth"):
        utils.trajectory_process_results(doc, ["[]"])
    with pytest.raises(ValueError):
        utils._points("[[1,2]]", 0, 100)
    doc = sample()
    doc["conversations"] = []
    with pytest.raises(ValueError):
        utils.share_robot_doc_to_target(doc)


def test_reversed_prediction_matches_reference_zero_iou() -> None:
    result = utils.affordance_process_results(sample(), ["[4,4,1,1]"])
    assert result["average_iou"] == 0
    assert result["valid_items"] == 1


def test_nontrivial_distances_match_pinned_reference() -> None:
    """Expected values come from FlagEvalMM commit 24655d2, not our implementation."""
    pred = np.array([[0.0, 0.0], [0.3, 0.1], [0.8, 0.7]])
    gt = np.array([[0.1, 0.2], [0.6, 0.4]])
    assert utils._distances(pred, gt) == pytest.approx((0.3605551275463989, 0.3605551275463989, 0.33911649915626346))


def test_overflowing_prediction_uses_origin_fallback(media: Path) -> None:
    result = utils.trajectory_process_results(sample(True), ["[[" + "9" * 400 + ", 1]]"])
    assert result["parse_valid_rate"] == 0
    assert np.isfinite(result["average_discrete_frechet_distance"])


@pytest.mark.parametrize(
    "box,iou,giou,pair",
    [
        ("[0,0,20,20]", 1, 1, (400, 400)),
        ("[0,0,10,20]", 0.5, 0.5, (200, 400)),
        ("[10,10,30,30]", 1 / 7, 1 / 7 - 2 / 9, (100, 700)),
        ("[20,0,40,20]", 0, 0, (0, 800)),
        ("[40,0,60,20]", 0, -1 / 3, (0, 800)),
    ],
)
def test_explicit_overlap_geometry(box: str, iou: float, giou: float, pair: tuple[float, float]) -> None:
    scores = utils.affordance_process_results(sample(), [box])
    assert scores["mean_iou"] == pytest.approx(iou)
    assert scores["mean_generalized_iou"] == pytest.approx(giou)
    assert scores["cumulative_iou"] == pytest.approx(pair)
    assert scores["precision_at_0_5"] == float(iou >= 0.5)
    assert scores["mean_generalized_iou"] <= scores["mean_iou"]


@pytest.mark.parametrize("answers", [[], [""], ["bad"], ["[1,2]"], ["[0,0,1e309,2]"], ["[4,4,1,1]"], ["[0,0,0,20]"], ["[0,0,1e200,1e200]"]])
def test_explicit_metrics_count_failures(answers: list[str]) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        scores = utils.affordance_process_results(sample(), answers)
    assert scores["mean_iou"] == scores["mean_generalized_iou"] == scores["precision_at_0_5"] == 0
    assert scores["cumulative_iou"] == (0, 400)


def test_explicit_aggregation_and_invalid_denominators() -> None:
    values = [(10.0, 10.0), (0.0, 90.0)]
    assert np.mean([i / u for i, u in values]) == 0.5
    assert utils.share_robot_cumulative_iou(values) == pytest.approx(0.1)
    # Gather worker payloads before taking the ratio, rather than averaging worker ratios.
    workers = [values[:1], values[1:]]
    assert utils.share_robot_cumulative_iou([pair for worker in workers for pair in worker]) == pytest.approx(0.1)
    assert utils.share_robot_cumulative_iou([]) == 0
    assert utils.share_robot_cumulative_iou([(1e308, 1e308)] * 2) == 1
    results = [utils.affordance_process_results(sample(), [answer]) for answer in ("[0,0,20,20]", "bad")]
    assert utils.share_robot_valid_mean([r["average_iou"] for r in results]) == 1
    assert np.mean([r["mean_iou"] for r in results]) == 0.5
    assert np.mean([r["precision_at_0_5"] for r in results]) == 0.5
    assert utils.share_robot_cumulative_iou([r["cumulative_iou"] for r in results]) == 0.5


@pytest.mark.parametrize("target", ["bad", "[0,0,0,2]", "[4,4,1,1]", "[0,0,1e309,2]", "[0,0,1e200,1e200]"])
def test_explicit_invalid_ground_truth(target: str) -> None:
    doc = sample()
    doc["conversations"][1]["value"] = target
    with pytest.raises(ValueError):
        utils.affordance_process_results(doc, ["[0,0,20,20]"])


@pytest.mark.parametrize("limits", [(200704, 1605632), (50176, 200704)])
@pytest.mark.parametrize("profile", ["qwen2_5_vl_chat", "qwen2_5_vl_simple"])
@pytest.mark.parametrize("size", [(300, 400), (399, 266), (1920, 1080), (28, 280), (448, 448)])
def test_qwen_dimensions_match_real_processor(profile: str, size: tuple[int, int], limits: tuple[int, int]) -> None:
    from qwen_vl_utils.vision_process import fetch_image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    min_pixels, max_pixels = limits
    settings = {} if profile.endswith("chat") else {"min_pixels": min_pixels, "max_pixels": max_pixels}
    intermediate = fetch_image({"image": Image.new("RGB", size), **settings})
    processor = Qwen2VLImageProcessor(size={"shortest_edge": min_pixels, "longest_edge": max_pixels})
    inputs = processor(images=[intermediate], return_tensors="pt")
    _, h, w = inputs["image_grid_thw"][0].tolist()
    assert utils._qwen_input_size(*size, profile, min_pixels, max_pixels) == (w * processor.patch_size, h * processor.patch_size)


def test_coordinate_scoring_preserves_raw_answer(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_qwen_input_size", lambda *args: (200, 400))
    answer = ["[0,0,40,40]"]
    kwargs = {"share_robot_coordinate_profile": "qwen2_5_vl_chat"}
    assert utils.affordance_process_results(sample(), answer, generation_kwargs=kwargs)["mean_iou"] == 1
    assert utils.affordance_process_results(sample(), answer)["mean_iou"] == 0.25
    assert answer == ["[0,0,40,40]"]
    assert utils.affordance_process_results(sample(), ["bad"], generation_kwargs=kwargs)["valid_items"] == 0
    with pytest.raises(ValueError, match="profile"):
        utils.affordance_process_results(sample(), answer, generation_kwargs={"share_robot_coordinate_profile": "unknown"})
    with pytest.raises(ValueError, match="limits"):
        utils.affordance_process_results(sample(), answer, generation_kwargs={**kwargs, "share_robot_min_pixels": -1})

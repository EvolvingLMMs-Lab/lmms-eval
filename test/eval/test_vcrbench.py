from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.vcrbench import utils as vcrbench_utils


@pytest.fixture(autouse=True)
def _isolate_media_env(monkeypatch, tmp_path):
    """Stop the media resolver from finding a developer's real video cache.

    media_resolver falls through VCRBENCH_ROOT -> LMMS_EVAL_MEDIA_ROOT -> HF_HOME.
    A shell that exports any of those makes the FileNotFoundError tests resolve a
    real file and silently stop testing anything. Clear them for every test and
    point HF_HOME at an empty tmp dir.
    """
    for var in ("VCRBENCH_ROOT", "VCRBENCH_VIDEO_DIR", "LMMS_EVAL_MEDIA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "_empty_hf_home"))


def _doc(ground_truth: Sequence[int] = (2, 0, 3, 1), goal: str = "make a pizza", qid: int = 7) -> dict[str, Any]:
    return {
        "qid": qid,
        "video_file": "video_7.mp4",
        "goal": goal,
        "ground_truth": list(ground_truth),
    }


def _write_test_video(path: Path, frame_count: int = 4, fps: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((8, 8, 3), value * 50, dtype=np.uint8))
    writer.release()


def test_vcrbench_task_is_registered() -> None:
    task_manager = TaskManager("ERROR")
    assert "vcrbench" in task_manager.all_tasks


def test_vcrbench_doc_to_visual_resolves_configured_video_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_dir = tmp_path / "flat_videos"
    video_dir.mkdir()
    _write_test_video(video_dir / "video_7.mp4")
    monkeypatch.setenv("VCRBENCH_VIDEO_DIR", str(video_dir))

    assert vcrbench_utils.vcrbench_doc_to_visual(_doc()) == [str(video_dir / "video_7.mp4")]


def test_vcrbench_doc_to_visual_resolves_root_with_videos_subdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    _write_test_video(video_dir / "video_7.mp4")
    monkeypatch.setenv("VCRBENCH_ROOT", str(tmp_path))

    assert vcrbench_utils.vcrbench_doc_to_visual(_doc()) == [str(video_dir / "video_7.mp4")]


def test_vcrbench_doc_to_visual_reports_the_missing_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VCRBENCH_VIDEO_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="VCRBench video not found"):
        vcrbench_utils.vcrbench_doc_to_visual(_doc())


def test_vcrbench_doc_to_text_substitutes_the_goal_into_both_slots() -> None:
    prompt = vcrbench_utils.vcrbench_doc_to_text(_doc(goal="make a pizza"))

    assert prompt.count("make a pizza") == 2
    assert "{goal}" not in prompt
    assert "Clip 1, Clip 2" in prompt
    assert prompt.endswith("Correct order: <mention the Clip numbers separated by a comma>\n")


def test_vcrbench_doc_to_text_wraps_the_prompt_with_task_kwargs() -> None:
    prompt = vcrbench_utils.vcrbench_doc_to_text(_doc(), {"pre_prompt": "Video:\n", "post_prompt": "\nOrder only."})

    assert prompt.startswith("Video:\nThe given video consists of multiple short clips")
    assert prompt.endswith("\nOrder only.")


def test_vcrbench_target_order_is_one_indexed() -> None:
    assert vcrbench_utils.vcrbench_target_order(_doc()) == [3, 1, 4, 2]
    assert vcrbench_utils.vcrbench_doc_to_target(_doc()) == "3, 1, 4, 2"


def test_vcrbench_extracts_a_boxed_permutation() -> None:
    assert vcrbench_utils.extract_predicted_order("Reasoning omitted. \\boxed{3, 1, 4, 2}", 4) == [3, 1, 4, 2]


def test_vcrbench_extracts_an_answer_tag_permutation() -> None:
    response = "<think>Clip 1 comes first.</think><answer>3, 1, 4, 2</answer>"

    assert vcrbench_utils.extract_predicted_order(response, 4) == [3, 1, 4, 2]


def test_vcrbench_extracts_the_official_correct_order_phrase() -> None:
    response = "First I identify each clip.\nCorrect order: 3, 1, 4, 2\n"

    assert vcrbench_utils.extract_predicted_order(response, 4) == [3, 1, 4, 2]


def test_vcrbench_extracts_a_clip_prefixed_order() -> None:
    assert vcrbench_utils.extract_predicted_order("Clip 3, Clip 1, Clip 4, Clip 2.", 4) == [3, 1, 4, 2]


def test_vcrbench_returns_zeros_for_a_length_mismatch() -> None:
    assert vcrbench_utils.extract_predicted_order("Correct order: 1, 2", 4) == [0, 0, 0, 0]


def test_vcrbench_returns_zeros_for_an_unparseable_answer() -> None:
    assert vcrbench_utils.extract_predicted_order("I cannot determine the order of the clips.", 4) == [0, 0, 0, 0]


def test_vcrbench_rejects_an_order_that_is_not_a_permutation() -> None:
    assert vcrbench_utils.extract_predicted_order("Correct order: 1, 1, 2, 3", 4) == [0, 0, 0, 0]


def test_vcrbench_compare_lists_scores_a_length_mismatch_as_target_length_zeros() -> None:
    assert vcrbench_utils.compare_lists([3, 1, 4, 2], [3, 1, 4, 2]) == [1, 1, 1, 1]
    assert vcrbench_utils.compare_lists([3, 1, 4, 2], [3, 2, 4, 1]) == [1, 0, 1, 0]
    assert vcrbench_utils.compare_lists([3, 1, 4, 2], [3, 1]) == [0, 0, 0, 0]


def test_vcrbench_process_results_reports_exact_match_and_step_scores() -> None:
    metrics = vcrbench_utils.vcrbench_process_results(_doc(), ["Correct order: 3, 2, 4, 1"])
    record = metrics["vcrbench_accuracy"]

    assert set(metrics) == {"vcrbench_accuracy", "vcrbench_step_accuracy", "vcrbench_weighted_accuracy"}
    assert record["qid"] == 7
    assert record["num_steps"] == 4
    assert record["answer"] == [3, 1, 4, 2]
    assert record["predicted_order"] == [3, 2, 4, 1]
    assert record["score"] == 0.0
    assert record["step_scores"] == [1, 0, 1, 0]


def test_vcrbench_process_results_scores_an_exact_order() -> None:
    record = vcrbench_utils.vcrbench_process_results(_doc(), ["Correct order: 3, 1, 4, 2"])["vcrbench_accuracy"]

    assert record["score"] == 1.0
    assert record["step_scores"] == [1, 1, 1, 1]


def test_vcrbench_process_results_zero_fills_an_unparseable_prediction() -> None:
    record = vcrbench_utils.vcrbench_process_results(_doc(), ["No idea."])["vcrbench_step_accuracy"]

    assert record["predicted_order"] == [0, 0, 0, 0]
    assert record["step_scores"] == [0, 0, 0, 0]


def _record(goal: str, score: float, step_scores: list[int]) -> dict[str, Any]:
    return {
        "qid": 0,
        "goal": goal,
        "num_steps": len(step_scores),
        "prediction": "",
        "predicted_order": [],
        "answer": [],
        "score": score,
        "step_scores": step_scores,
    }


def test_vcrbench_aggregate_accuracy_is_the_exact_match_rate() -> None:
    results = [_record("pizza", 1.0, [1, 1]), _record("pizza", 0.0, [1, 0]), _record("tea", 1.0, [1, 1])]

    assert vcrbench_utils.vcrbench_aggregate_accuracy(results) == pytest.approx(200.0 / 3.0)
    assert vcrbench_utils.vcrbench_aggregate_accuracy([]) == 0.0


def test_vcrbench_aggregate_step_accuracy_pools_every_clip_position() -> None:
    results = [_record("pizza", 1.0, [1, 1]), _record("pizza", 0.0, [1, 0]), _record("tea", 1.0, [1, 1])]

    assert vcrbench_utils.vcrbench_aggregate_step_accuracy(results) == pytest.approx(500.0 / 6.0)
    assert vcrbench_utils.vcrbench_aggregate_step_accuracy([]) == 0.0


def test_vcrbench_aggregate_weighted_accuracy_averages_over_the_goal_classes() -> None:
    results = [_record("pizza", 1.0, [1, 1]), _record("pizza", 0.0, [1, 0]), _record("tea", 1.0, [1, 1])]

    assert vcrbench_utils.vcrbench_aggregate_weighted_accuracy(results) == pytest.approx(75.0)
    assert vcrbench_utils.vcrbench_aggregate_weighted_accuracy([]) == 0.0

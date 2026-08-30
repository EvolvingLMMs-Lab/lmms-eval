from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.longvideo_reason import utils as lvr_utils

OPTIONS = "A. First option.\nB. Second option.\nC. Third option.\nD. Fourth option."


@pytest.fixture(autouse=True)
def _isolate_media_env(monkeypatch, tmp_path):
    """Stop the media resolver from finding a developer's real video cache.

    media_resolver falls through LONGVIDEO_REASON_ROOT -> LMMS_EVAL_MEDIA_ROOT
    -> HF_HOME. A shell exporting any of those makes the FileNotFoundError tests
    resolve a real file and silently stop testing anything.
    """
    for var in ("LONGVIDEO_REASON_ROOT", "LONGVIDEO_REASON_VIDEO_DIR", "LMMS_EVAL_MEDIA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "_empty_hf_home"))


def _doc(
    answer: str = "<answer>B</answer>",
    problem_type: str = "goal",
    problem: str | None = None,
    problem_id: int = 0,
    videos: str = "longvila_videos/clip.mp4",
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "problem": problem if problem is not None else f"What happens next?\n{OPTIONS}",
        "data_type": "video",
        "problem_type": problem_type,
        "reasoning": "irrelevant to scoring",
        "videos": videos,
        "answer": answer,
    }


def _write_test_video(path: Path, frame_count: int = 4, fps: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((8, 8, 3), value * 50, dtype=np.uint8))
    writer.release()


# --------------------------------------------------------------------------
# Registration and prompt
# --------------------------------------------------------------------------


def test_task_is_registered() -> None:
    task_manager = TaskManager("ERROR")
    assert "longvideo_reason" in task_manager.all_tasks


def test_doc_to_text_uses_the_official_template_verbatim() -> None:
    text = lvr_utils.longvideo_reason_doc_to_text(_doc())
    assert text.startswith("You are a helpful assistant. The user asks a question, and then you solves it.")
    assert "<think> </think> and <answer> </answer> tags" in text
    # The options live inside `problem`; the template must not rebuild them.
    assert text.endswith("A. First option.\nB. Second option.\nC. Third option.\nD. Fourth option.")


def test_doc_to_text_applies_pre_and_post_prompt() -> None:
    text = lvr_utils.longvideo_reason_doc_to_text(_doc(), {"pre_prompt": "<PRE>", "post_prompt": "<POST>"})
    assert text.startswith("<PRE>You are a helpful assistant.")
    assert text.endswith("<POST>")


def test_doc_to_target_returns_the_wrapped_gold() -> None:
    assert lvr_utils.longvideo_reason_doc_to_target(_doc()) == "<answer>B</answer>"


# --------------------------------------------------------------------------
# Media resolution
# --------------------------------------------------------------------------


def test_doc_to_visual_resolves_parent_of_longvila_videos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_dir = tmp_path / "longvila_videos"
    video_dir.mkdir()
    _write_test_video(video_dir / "clip.mp4")
    monkeypatch.setenv("LONGVIDEO_REASON_VIDEO_DIR", str(tmp_path))

    assert lvr_utils.longvideo_reason_doc_to_visual(_doc()) == [str(video_dir / "clip.mp4")]


def test_doc_to_visual_resolves_a_flat_video_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The extracted tree is sometimes flattened; the basename must still resolve."""
    video_dir = tmp_path / "flat"
    video_dir.mkdir()
    _write_test_video(video_dir / "clip.mp4")
    monkeypatch.setenv("LONGVIDEO_REASON_VIDEO_DIR", str(video_dir))

    assert lvr_utils.longvideo_reason_doc_to_visual(_doc()) == [str(video_dir / "clip.mp4")]


@pytest.mark.parametrize("extension", ["mp4", "webm", "mkv"])
def test_doc_to_visual_resolves_every_shipped_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extension: str) -> None:
    """The split is 867 .mp4, 130 .webm and 3 .mkv, so a resolver that only
    understands .mp4 loses 13% of the benchmark."""
    video_dir = tmp_path / "longvila_videos"
    video_dir.mkdir()
    (video_dir / f"clip.{extension}").write_bytes(b"\x00")
    monkeypatch.setenv("LONGVIDEO_REASON_VIDEO_DIR", str(tmp_path))

    resolved = lvr_utils.longvideo_reason_doc_to_visual(_doc(videos=f"longvila_videos/clip.{extension}"))
    assert resolved == [str(video_dir / f"clip.{extension}")]


def test_doc_to_visual_resolves_the_per_shard_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ten tar.gz shards extract into "longvideo_eval_subset<N>/", NOT into
    the "longvila_videos/" directory the upstream README documents. A resolver
    that only knows the documented layout reports a complete 195 GB download as
    1,000 missing videos."""
    video_dir = tmp_path / "longvideo_eval_subset7"
    video_dir.mkdir()
    _write_test_video(video_dir / "clip.mp4")
    monkeypatch.setenv("LONGVIDEO_REASON_VIDEO_DIR", str(tmp_path))

    assert lvr_utils.longvideo_reason_doc_to_visual(_doc()) == [str(video_dir / "clip.mp4")]


def test_doc_to_visual_raises_when_the_video_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LONGVIDEO_REASON_VIDEO_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="longvideo_eval_subset"):
        lvr_utils.longvideo_reason_doc_to_visual(_doc())


# --------------------------------------------------------------------------
# Official extraction (paper parity)
# --------------------------------------------------------------------------


def test_official_extraction_returns_the_answer_span() -> None:
    assert lvr_utils.extract_official_answer("<think>x</think> <answer>C</answer>") == "C"


def test_official_extraction_handles_the_phrase_branch() -> None:
    """The official parser strips "Therefore the final answer is: " when present."""
    completion = "<think>x</think> <answer>Therefore the final answer is: D</answer>"
    assert lvr_utils.extract_official_answer(completion) == "D"


def test_official_extraction_falls_back_to_the_whole_completion() -> None:
    assert lvr_utils.extract_official_answer("  B  ") == "B"


def test_strict_score_is_byte_exact() -> None:
    """A trailing period misses under the official comparison. This is upstream
    behaviour and the strict metric must reproduce it, not repair it."""
    record = lvr_utils.longvideo_reason_process_results(_doc(), ["<think>x</think> <answer>B.</answer>"])
    assert record["longvideo_reason_strict_accuracy"]["strict_score"] == 0.0


def test_strict_score_matches_a_clean_answer() -> None:
    record = lvr_utils.longvideo_reason_process_results(_doc(), ["<think>x</think> <answer>B</answer>"])
    assert record["longvideo_reason_strict_accuracy"]["strict_score"] == 1.0


# --------------------------------------------------------------------------
# Robust extraction (additive layer)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion",
    [
        "<think>x</think> <answer>B</answer>",
        "<think>x</think> <answer>B.</answer>",
        "<think>x</think> <answer>\\boxed{B}</answer>",
        "<think>x</think><answer>(B)</answer>",
        "The correct answer is B.",
        "\\boxed{B}",
        "B",
    ],
)
def test_robust_extraction_recovers_common_answer_forms(completion: str) -> None:
    assert lvr_utils.extract_longvideo_reason_answer(completion, ["A", "B", "C", "D"]) == "B"


def test_robust_extraction_never_downgrades_an_official_hit() -> None:
    """The additive property the two metrics depend on: whenever the official
    parser yields an offered letter, the robust parser returns the SAME letter."""
    for letter in ("A", "B", "C", "D"):
        completion = f"<think>irrelevant mention of A and C</think> <answer>{letter}</answer>"
        assert lvr_utils.extract_longvideo_reason_answer(completion, ["A", "B", "C", "D"]) == letter


def test_robust_extraction_rejects_a_letter_not_offered() -> None:
    """A row with only two options must never be scored against a letter it does
    not offer."""
    assert lvr_utils.extract_longvideo_reason_answer("<answer>D</answer>", ["A", "B"]) != "D"


# --------------------------------------------------------------------------
# Malformed rows (the 5 contaminated examples)
# --------------------------------------------------------------------------


def test_wellformed_row_is_flagged_wellformed() -> None:
    record = lvr_utils.longvideo_reason_process_results(_doc(), ["<answer>B</answer>"])
    assert record["longvideo_reason_overall_accuracy"]["wellformed"] is True


def test_row_with_no_options_is_flagged_malformed() -> None:
    """problem_id 379 and 857 ship with no option block at all."""
    doc = _doc(problem="Who is the man interacting with the car?\n\n", problem_id=379)
    record = lvr_utils.longvideo_reason_process_results(doc, ["<answer>C</answer>"])
    assert record["longvideo_reason_overall_accuracy"]["wellformed"] is False


def test_row_with_leaked_scratchpad_is_flagged_malformed() -> None:
    """problem_id 147, 743 and 825 carry generator deliberation where an option
    should be, so fewer than four letters parse."""
    doc = _doc(
        problem="What was the primary factor?\nC. But the user wants a question that requires multiple steps.",
        problem_id=743,
    )
    record = lvr_utils.longvideo_reason_process_results(doc, ["<answer>C</answer>"])
    assert record["longvideo_reason_overall_accuracy"]["wellformed"] is False


def test_wellformed_aggregate_excludes_malformed_rows() -> None:
    good = lvr_utils.longvideo_reason_process_results(_doc(), ["<answer>B</answer>"])["longvideo_reason_overall_accuracy"]
    bad = lvr_utils.longvideo_reason_process_results(
        _doc(problem="No options here.\n\n", problem_id=379, answer="<answer>C</answer>"),
        ["<answer>A</answer>"],
    )["longvideo_reason_overall_accuracy"]

    assert lvr_utils.longvideo_reason_aggregate_overall([good, bad]) == 50.0
    assert lvr_utils.longvideo_reason_aggregate_wellformed([good, bad]) == 100.0


# --------------------------------------------------------------------------
# Format metric and per-perspective aggregation
# --------------------------------------------------------------------------


def test_format_score_requires_think_then_answer_from_the_start() -> None:
    matching = lvr_utils.longvideo_reason_process_results(_doc(), ["<think>x</think> <answer>B</answer>"])
    assert matching["longvideo_reason_format_accuracy"]["format_score"] == 1.0

    prefixed = lvr_utils.longvideo_reason_process_results(_doc(), ["Sure! <think>x</think> <answer>B</answer>"])
    assert prefixed["longvideo_reason_format_accuracy"]["format_score"] == 0.0


def test_process_results_emits_the_perspective_metric() -> None:
    for problem_type in lvr_utils.PROBLEM_TYPES:
        record = lvr_utils.longvideo_reason_process_results(_doc(problem_type=problem_type), ["<answer>B</answer>"])
        assert f"longvideo_reason_{problem_type}_accuracy" in record


def test_process_results_rejects_an_unknown_perspective() -> None:
    with pytest.raises(ValueError, match="Unknown LongVideo-Reason problem_type"):
        lvr_utils.longvideo_reason_process_results(_doc(problem_type="narrative"), ["<answer>B</answer>"])


def test_perspective_aggregate_selects_only_its_own_rows() -> None:
    temporal_hit = lvr_utils.longvideo_reason_process_results(_doc(problem_type="temporal"), ["<answer>B</answer>"])["longvideo_reason_overall_accuracy"]
    spatial_miss = lvr_utils.longvideo_reason_process_results(_doc(problem_type="spatial"), ["<answer>A</answer>"])["longvideo_reason_overall_accuracy"]
    records = [temporal_hit, spatial_miss]

    assert lvr_utils.longvideo_reason_aggregate_temporal(records) == 100.0
    assert lvr_utils.longvideo_reason_aggregate_spatial(records) == 0.0
    assert lvr_utils.longvideo_reason_aggregate_overall(records) == 50.0


def test_aggregate_over_no_records_is_zero() -> None:
    assert lvr_utils.longvideo_reason_aggregate_goal([]) == 0.0


def test_gold_is_unwrapped_before_comparison() -> None:
    """The gold field ships as "<answer>B</answer>", not as a bare letter."""
    record = lvr_utils.longvideo_reason_process_results(_doc(answer="<answer>B</answer>"), ["<answer>B</answer>"])
    assert record["longvideo_reason_overall_accuracy"]["answer"] == "B"
    assert record["longvideo_reason_overall_accuracy"]["score"] == 1.0

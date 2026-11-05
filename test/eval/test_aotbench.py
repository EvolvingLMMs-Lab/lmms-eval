import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.aotbench import utils as aotbench_utils

SUBTASKS = ["aotbench_reversefilm", "aotbench_ucf101", "aotbench_rtime_t2v", "aotbench_rtime_v2t", "aotbench_qa"]


def _doc(**extra):
    doc = {"question": "Which segment?\nA. First\nB. Second", "ans": "B", "video_name": "videos/demo/clip.mp4"}
    doc.update(extra)
    return doc


def _score(reply, **extra):
    return aotbench_utils.aotbench_process_results(_doc(**extra), [reply])["aotbench_acc"]


def test_aotbench_tasks_are_registered():
    task_manager = TaskManager()
    assert "aotbench" in task_manager.all_groups
    assert set(SUBTASKS) <= set(task_manager.all_subtasks)


def test_doc_to_text_returns_question_unchanged():
    assert aotbench_utils.aotbench_doc_to_text(_doc()) == _doc()["question"]


@pytest.mark.parametrize(
    "reply, correct",
    [("B", True), ("B.", True), ("B. Second", True), (" B\n", True), ("A", False), ("The answer is B", False), ("b", False), ("", False), ("   ", False)],
)
def test_only_the_first_character_is_scored(reply, correct):
    assert _score(reply)["correct"] is correct


def test_aggregate_is_overall_accuracy():
    results = [_score("B", question_type="nextqa"), _score("A", question_type="nextqa"), _score("B", question_type="tempcompass"), _score("B", question_type="tempcompass")]
    assert aotbench_utils.aotbench_aggregate(results) == 0.75


def test_aggregate_without_source_field():
    assert aotbench_utils.aotbench_aggregate([_score("B"), _score("A")]) == 0.5


def test_doc_to_visual_resolves_under_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    video = tmp_path / "aotbench" / "videos" / "demo" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    assert aotbench_utils.aotbench_doc_to_visual(_doc()) == [str(video)]


def test_doc_to_visual_reports_missing_video(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        aotbench_utils.aotbench_doc_to_visual(_doc())

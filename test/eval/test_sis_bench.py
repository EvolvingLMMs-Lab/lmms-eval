import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.sis_bench import utils


def _doc(task_type="action_recognition", answer="A"):
    return {
        "question_id": f"{task_type}_0001",
        "video_name": "AirScape/AirScape_0001.mp4",
        "video_path": "UAVideo/AirScape/AirScape_0001.mp4",
        "concat_num": 1,
        "task_type": task_type,
        "question": "What is the drone doing?",
        "options": {"A": "Ascending", "B": "Descending", "C": "Hovering", "D": "Turning"},
        "answer": answer,
    }


def test_sis_bench_is_registered():
    assert "sis_bench" in TaskManager("WARNING").all_subtasks


def test_doc_to_visual_uses_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    video = tmp_path / "sis_bench" / "video" / "AirScape" / "AirScape_0001.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    assert utils.sis_bench_doc_to_visual(_doc()) == [str(video)]


def test_doc_to_visual_rejects_path_traversal():
    doc = _doc()
    doc["video_name"] = "../secret.mp4"
    with pytest.raises(ValueError, match="must be relative"):
        utils.sis_bench_doc_to_visual(doc)


def test_doc_to_text_formats_all_options():
    prompt = utils.sis_bench_doc_to_text(_doc(), {"pre_prompt": "Video QA:\n", "post_prompt": "Letter only."})
    assert prompt.startswith("Video QA:\nWhat is the drone doing?")
    assert "(A) Ascending" in prompt
    assert "(D) Turning" in prompt
    assert prompt.endswith("Letter only.")


def test_process_results_and_aggregations():
    self_aware = utils.sis_bench_process_results(_doc(), ["The answer is (A)."])
    spatial = utils.sis_bench_process_results(_doc("object_existence", answer="B"), ["C"])
    records = [self_aware["sis_bench_overall_accuracy"], spatial["sis_bench_overall_accuracy"]]

    assert set(self_aware) == {
        "sis_bench_overall_accuracy",
        "sis_bench_self_awareness_accuracy",
        "sis_bench_action_recognition_accuracy",
    }
    assert set(spatial) == {
        "sis_bench_overall_accuracy",
        "sis_bench_spatial_cognition_accuracy",
        "sis_bench_object_existence_accuracy",
    }
    assert records[0]["predicted_answer"] == "A"
    assert records[0]["score"] == 1
    assert records[1]["score"] == 0
    assert utils.sis_bench_aggregate_overall(records) == 50.0
    assert utils.sis_bench_aggregate_spatial_cognition(records) == 0.0
    assert utils.sis_bench_aggregate_self_awareness(records) == 100.0


def test_dimension_accuracy_is_weighted_by_question_count():
    records = [
        utils.sis_bench_process_results(_doc("object_existence", answer="A"), ["A"])["sis_bench_spatial_cognition_accuracy"],
        utils.sis_bench_process_results(_doc("object_existence", answer="A"), ["A"])["sis_bench_spatial_cognition_accuracy"],
        utils.sis_bench_process_results(_doc("relative_direction", answer="A"), ["B"])["sis_bench_spatial_cognition_accuracy"],
    ]

    assert utils.sis_bench_aggregate_spatial_cognition(records) == pytest.approx(100 * 2 / 3)


def test_per_task_accuracy():
    correct = utils.sis_bench_process_results(_doc("relative_direction", answer="A"), ["A"])
    wrong = utils.sis_bench_process_results(_doc("relative_direction", answer="A"), ["B"])
    records = [correct["sis_bench_relative_direction_accuracy"], wrong["sis_bench_relative_direction_accuracy"]]

    assert utils.sis_bench_aggregate_relative_direction(records) == 50.0


def test_unknown_task_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown SIS-Bench task type"):
        utils.sis_bench_process_results(_doc("unknown"), ["A"])

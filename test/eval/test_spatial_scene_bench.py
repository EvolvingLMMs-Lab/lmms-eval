import pytest
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.spatial_scene_bench import utils


def _doc(question_type="count", answer_type="count", answer="3", options=()):
    return {
        "id": "scene00000_q001",
        "question_type": question_type,
        "question": "How many cubes are there?",
        "answer": answer,
        "answer_type": answer_type,
        "options": list(options),
        "images": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
    }


def test_spatial_scene_bench_tasks_are_registered():
    task_manager = TaskManager("ERROR")
    assert {"spatial_scene_bench", "spatial_scene_bench_mini"} <= set(task_manager.all_tasks)


@pytest.mark.parametrize(
    "text,answer_type,options,expected",
    [
        ("3", "count", None, "3"),
        ("There are three cubes.", "count", None, "3"),
        ("<answer>2.0</answer>", "count", None, "2"),
        ("1.25 meters", "numeric", None, "1.25"),
        ("Yes, it is.", "yesno", None, "yes"),
        ("(C)", "choice", ["a", "b", "c", "d"], "C"),
        ("The answer is B.", "choice", ["a", "b", "c", "d"], "B"),
        ("blue cube", "choice", ["red sphere", "blue cube", "gray cylinder", "metal cube"], "B"),
        ("I am not sure", "count", None, None),
    ],
)
def test_parse_answer(text, answer_type, options, expected):
    assert utils.parse_answer(text, answer_type, options) == expected


def test_mean_relative_accuracy():
    assert utils.mean_relative_accuracy(2.0, 2.0) == 1.0
    assert utils.mean_relative_accuracy(2.2, 2.0) == pytest.approx(0.8)
    assert utils.mean_relative_accuracy(10.0, 2.0) == 0.0
    assert utils.mean_relative_accuracy(0.0, 0.0) == 1.0


def test_messages_interleave_all_frames_before_prompt():
    messages = utils.ssb_doc_to_messages(_doc())
    content = messages[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[-1]["text"].endswith("Answer with a single integer.")


def test_process_results_and_aggregation():
    rows = [
        utils.ssb_process_results(_doc(answer="3"), ["3"]),
        utils.ssb_process_results(_doc(answer="4"), ["3"]),
        utils.ssb_process_results(_doc("video_distance", "numeric", "2.0"), ["2.0"]),
        utils.ssb_process_results(_doc("relation", "yesno", "no"), ["yes"]),
    ]
    payloads = [r["ssb_overall"] for r in rows]

    assert utils.ssb_aggregate_count(payloads) == 50.0
    assert utils.ssb_aggregate_video_distance(payloads) == 100.0
    assert utils.ssb_aggregate_overall(payloads) == pytest.approx((50.0 + 100.0 + 0.0) / 3)
    assert utils.ssb_aggregate_image_overall(payloads) == 25.0
    assert utils.ssb_aggregate_video_overall(payloads) == 100.0
    assert utils.ssb_aggregate_count_mra(payloads) == pytest.approx(100 * (1.0 + 0.5) / 2)

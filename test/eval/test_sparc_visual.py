import json
import math

import datasets
import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.sparc_visual import utils

# The worked example from the SPaRC prompt, with its published solution.
SOLUTION = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 6), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4), (6, 3), (6, 2), (5, 2), (4, 2), (4, 1), (4, 0), (5, 0)]


def _doc(level=3):
    return {
        "id": "example",
        "difficulty_level": level,
        "difficulty_score": 2.5,
        "grid_size": {"width": 3, "height": 3},
        "polyshapes": json.dumps({"112": [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], "624": [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]}),
        "puzzle_array": [
            ["+", ".", "+", "+", "+", "E", "+"],
            ["+", "C-R", "+", "o-K", "+", "o-K", "+"],
            ["S", "+", "+", "+", "+", "+", "+"],
            ["+", "P-G-112", "+", "*-G", "+", "P-B-624", "+"],
            ["+", "+", "+", "+", "+", "+", "+"],
            ["+", "*-G", "+", "*-G", "+", "o-K", "+"],
            ["+", "+", "+", ".", "+", "+", "+"],
        ],
        "solution_count": 1,
        "solutions": [{"index": 0, "path": [{"x": x, "y": y} for x, y in SOLUTION]}],
        "text_visualization": "puzzle:\n  grid_size:\n    width: 3\n    height: 3\n",
    }


def _answer(path):
    return "Step-by-step reasoning ...\n#### " + ", ".join(f"({x},{y})" for x, y in path)


def test_sparc_visual_tasks_are_registered():
    task_manager = TaskManager("ERROR")
    assert {"sparc_visual", "sparc_visual_hard"} <= set(task_manager.all_tasks)


def test_gold_solution_scores_one():
    doc = _doc()
    for answer in (_answer(SOLUTION), "####\n[" + ", ".join(f"({x}, {y})" for x, y in SOLUTION) + "]", utils.sparc_visual_doc_to_target(doc)):
        assert utils.sparc_visual_process_results(doc, [answer]) == {"acc": 1.0, "invalid_path": 0.0}


@pytest.mark.parametrize(
    "answer,invalid",
    [
        (_answer(SOLUTION[::-1]), 1.0),  # wrong start and end
        (_answer(SOLUTION[:-1]), 1.0),  # does not reach the end node
        (_answer([(0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]), 1.0),  # crosses rule cell (1, 1)
        (_answer([(0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]), 0.0),  # valid line, breaks the rules
        ("I cannot solve this puzzle.", 1.0),
        ("", 1.0),
    ],
)
def test_wrong_answers_score_zero(answer, invalid):
    assert utils.sparc_visual_process_results(_doc(), [answer]) == {"acc": 0.0, "invalid_path": invalid}


def test_extraction_reads_after_last_marker_and_stops_at_end_node():
    answer = "#### (9,9)\nfinal answer\n" + _answer(SOLUTION) + ", (6,0), (6,1)"
    path = utils.extract_solution_path(answer, _doc())
    assert [(p["x"], p["y"]) for p in path] == SOLUTION


def test_per_level_accuracy():
    rows = [
        utils.sparc_visual_process_results_by_level(_doc(level=1), [_answer(SOLUTION)]),
        utils.sparc_visual_process_results_by_level(_doc(level=1), ["no path"]),
        utils.sparc_visual_process_results_by_level(_doc(level=2), [_answer(SOLUTION)]),
    ]
    assert [r["acc"] for r in rows] == [1.0, 0.0, 1.0]
    payloads = [r["acc_level_1"] for r in rows]
    assert all(r[f"acc_level_{level}"] == r["acc_level_1"] for r in rows for level in utils.LEVELS)
    assert utils.sparc_visual_aggregate_level_1(payloads) == 0.5
    assert utils.sparc_visual_aggregate_level_2(payloads) == 1.0
    assert math.isnan(utils.sparc_visual_aggregate_level_5(payloads))


def test_round_robin_order_and_hard_subset():
    dataset = datasets.Dataset.from_dict({"id": list("abcdefg"), "difficulty_level": [1, 1, 1, 2, 3, 5, 5]})
    assert utils.sparc_visual_process_docs(dataset)["id"] == ["a", "d", "e", "f", "b", "g", "c"]
    assert utils.sparc_visual_process_hard(dataset)["id"] == ["f", "g"]


def test_prompt_contains_puzzle_text():
    doc = _doc()
    text = utils.sparc_visual_doc_to_text(doc)
    assert text.startswith("You are an expert spatial reasoning AI specializing in solving puzzles from the game 'The Witness'. \nYour task")
    assert "grid of size 6x6" in text
    assert "Shape 624:\n[0, 1, 0, 0]\n[0, 1, 1, 0]" in text
    assert doc["text_visualization"] in text
    assert text.endswith('after this string: "####". DON\'T SKIP ANY intermediate nodes (the distance between each node must be 1).\nExample coordinate list: [(0,0), (1,0), (2,0), (2,1), ...]\n')


def test_messages_put_system_prompt_first_and_image_before_text():
    pytest.importorskip("sparc_visualization")
    doc = _doc()
    messages = utils.sparc_visual_doc_to_messages(doc)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == [{"type": "text", "text": utils.SYSTEM_PROMPT}]
    image, text = messages[1]["content"]
    assert image["type"] == "image" and image["url"].mode == "RGB"
    assert image["url"].size == (560, 560)
    assert text == {"type": "text", "text": utils.sparc_visual_doc_to_text(doc)}

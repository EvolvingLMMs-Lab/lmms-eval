import json

import pytest

from lmms_eval.tasks.arc_agi_1 import utils as arc_agi_1_utils
from lmms_eval.tasks.arc_agi_2 import utils as arc_agi_2_utils

ANSWER = [[1, 0], [0, 1]]
TEST_INPUT = [[0, 1], [1, 0]]

PARSERS = [
    pytest.param(arc_agi_1_utils._parse_grid_from_response, id="arc_agi_1"),
    pytest.param(arc_agi_2_utils._parse_grid_from_response, id="arc_agi_2"),
]


@pytest.mark.parametrize("parse", PARSERS)
@pytest.mark.parametrize(
    "response",
    [
        pytest.param(json.dumps(ANSWER), id="bare grid"),
        pytest.param(f"```json\n{json.dumps(ANSWER)}\n```", id="fenced grid"),
        pytest.param("```json\n[\n  [1, 0],\n  [0, 1]\n]\n```", id="multi-line fenced grid"),
        pytest.param(f"The output is \\boxed{{{json.dumps(ANSWER)}}}", id="boxed grid"),
        pytest.param(f"The test input is {json.dumps(TEST_INPUT)}. Swapping the colors gives {json.dumps(ANSWER)}.", id="input quoted first"),
        pytest.param(f"Input:\n```json\n{json.dumps(TEST_INPUT)}\n```\nOutput:\n```json\n{json.dumps(ANSWER)}\n```", id="input and output fenced"),
        pytest.param(f"Input:\n```json\n{json.dumps(TEST_INPUT)}\n```\nSo the output is {json.dumps(ANSWER)}", id="fenced input, bare answer"),
        pytest.param(f"My first guess was {json.dumps(TEST_INPUT)}, but checking the examples again the output is:\n{json.dumps(ANSWER)}", id="corrected answer"),
        pytest.param(
            f"For the test input `{json.dumps(TEST_INPUT)}`:\nthe colors are swapped.\n\nThus, the output is:\n\n```python\n{json.dumps(ANSWER)}\n```\n\n\\boxed{{{json.dumps(ANSWER)}}}",
            id="official harness style",
        ),
    ],
)
def test_parse_grid_uses_the_last_grid(parse, response):
    assert parse(response) == ANSWER


@pytest.mark.parametrize("parse", PARSERS)
def test_parse_grid_returns_none_without_a_grid(parse):
    assert parse("I could not find the pattern.") is None


def test_arc_agi_1_scores_the_final_grid():
    response = f"The test input is {json.dumps(TEST_INPUT)}, so the output is {json.dumps(ANSWER)}."
    doc = {"id": "x", "raw_solution": [ANSWER]}

    assert arc_agi_1_utils.arc_agi_1_process_results(doc, [response]) == {"arc_agi_1_acc": 1.0}


def test_arc_agi_2_scores_the_final_grid():
    response = f"The test input is {json.dumps(TEST_INPUT)}, so the output is {json.dumps(ANSWER)}."
    doc = {"id": "x", "test_outputs": [ANSWER]}

    assert arc_agi_2_utils.arc_agi_2_process_results(doc, [response])["arc_agi_2_acc"] == 1.0

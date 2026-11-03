"""CharXiv reasoning instructions and grading follow answer type, not source."""

from types import ModuleType
from typing import Any

import pytest

from lmms_eval.tasks.charxiv import utils
from lmms_eval.tasks.charxiv.reasoning import utils as cot_utils

# Real princeton-nlp/CharXiv validation rows. In the official annotations,
# reasoning_a_type is inst_category; reasoning_q_source is qa_source.
REASONING_CASES = [
    (
        "images/2.jpg",
        "What is the name of the line that is the furthest away from its fi value from the W_H axis?",
        "lambda_L = 0.13",
        1,
        3,
        "grounded to some text that is explicitly written",
    ),
    (
        "images/21.jpg",
        "What are the categories with the two largest clusters by area?",
        "case, det",
        2,
        3,
        "you are allowed to respond with a short phrase only",
    ),
    (
        "images/14.jpg",
        "Provide the accuracy (%) of the model with the second highest time.",
        "94",
        3,
        2,
        "grounded to a number that is exlicitly written",
    ),
    (
        "images/17.jpg",
        "How many data points have a DAVIS-2017 J & F Mean score of less than 50?",
        "3",
        4,
        2,
        "Your final answer must be an exact integer.",
    ),
    (
        "images/61.jpg",
        r"How much does the accuracy decrease for Class pair 3 and 8 from no noise to \( \epsilon = 1.22 \)?",
        "9.17",
        4,
        1,
        "Your final answer must be a number with 2 decimal places.",
    ),
]


@pytest.fixture(params=REASONING_CASES, ids=["text-in-chart", "text-in-general", "number-in-chart", "integer-in-general", "decimal-in-general"])
def reasoning_case(request: pytest.FixtureRequest) -> tuple[dict[str, Any], str]:
    """Return an annotated row and an independently specified instruction."""
    figure_path, question, answer, answer_type, source, instruction = request.param
    doc = {
        "figure_path": figure_path,
        "reasoning_q": question,
        "reasoning_a": answer,
        "reasoning_a_type": answer_type,
        "reasoning_q_source": source,
    }
    return doc, instruction


@pytest.mark.parametrize("task_utils", [utils, cot_utils], ids=["reasoning", "reasoning-cot"])
def test_reasoning_instruction_uses_answer_type(reasoning_case: tuple[dict[str, Any], str], task_utils: ModuleType) -> None:
    doc, instruction = reasoning_case

    prompt = task_utils.charxiv_reasoning_doc_to_text_cot(doc)

    assert prompt.startswith(doc["reasoning_q"])
    assert instruction in prompt


def test_reasoning_grading_uses_answer_type(reasoning_case: tuple[dict[str, Any], str]) -> None:
    doc, _ = reasoning_case

    result = utils.charxiv_reasoning_process_results(doc, [doc["reasoning_a"]])

    assert result["reasoning_acc"]["data"]["inst_category"] == doc["reasoning_a_type"]

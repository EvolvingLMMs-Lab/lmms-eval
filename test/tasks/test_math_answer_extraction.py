"""Final-answer extraction for the reasoning tasks, OlympiadBench (boxed) and the MathVision prompt.

Each case below scored wrong before the fix; the values were measured, not assumed.
"""

import importlib
import sys

import pytest

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score, extract_anwser_tag
from lmms_eval.tasks.mathvision.utils import mathvision_doc_to_text

# --- extract_anwser_tag: the answer after the reasoning, not a tag mentioned inside it ---


def test_answer_tag_mentioned_while_thinking_does_not_win():
    response = "<think>I must put the result in <answer></answer> tags.</think><analysis>2+1</analysis><answer>3</answer>"
    assert extract_anwser_tag(response) == "3"


def test_empty_answer_tag_while_thinking_falls_through_to_the_box():
    response = "<think>Format: <answer></answer>. Compute 1+2.</think>So the result is \\boxed{3}"
    assert extract_anwser_tag(response) == "3"


def test_box_drafted_while_thinking_does_not_beat_the_final_answer():
    response = "<think>Maybe \\boxed{2}? No, recount.</think>The count is 3."
    assert extract_anwser_tag(response) == "3"


def test_last_non_empty_answer_tag_wins():
    assert extract_anwser_tag("<answer>2</answer> Wait, I misread the axis. <answer>3</answer>") == "3"
    assert extract_anwser_tag("<answer>3</answer><answer> </answer>") == "3"


def test_truncated_thinking_keeps_the_old_behaviour():
    # No closing </think>: nothing is stripped.
    assert extract_anwser_tag("<think>The total is 7") == "7"
    assert extract_anwser_tag("<answer>B</answer>") == "B"


# --- extract_anwser_tag: the number fallback keeps the sign ---


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Thus x = -3", "-3"),
        ("The slope is -2.5.", "-2.5"),
        ("x=-4", "-4"),
        ("The answer is 204", "204"),
        ("It covers pages 2-3", "3"),
    ],
)
def test_number_fallback_keeps_a_leading_minus_sign(response, expected):
    assert extract_anwser_tag(response) == expected


def test_compute_score_no_longer_credits_a_dropped_sign():
    assert compute_score("test", "Thus x = -3", "3")["acc_score"] == 0.0
    assert compute_score("test", "Thus x = -3", "-3")["acc_score"] == 1.0


# --- MathVision prompt: the instruction and the question are separate lines ---


def test_mathvision_prompt_separates_instruction_and_question():
    prompt = mathvision_doc_to_text({"question": "What is 2+2?", "options": []})
    assert prompt == 'Please solve the problem step by step and put your answer in one "\\boxed{}".\nWhat is 2+2?'
    prompt = mathvision_doc_to_text({"question": "Pick one.", "options": ["1", "2"]}, {"mc_prompt": "Answer with the letter."})
    assert prompt.endswith('"\\boxed{}".\nPick one.\nChoices: A. 1\nB. 2\nAnswer with the letter.')


# --- OlympiadBench (boxed): the final box, not the first draft ---


@pytest.fixture
def mimo_utils(monkeypatch):
    from lmms_eval.tasks.olympiadbench_mimo import olympiadbench_evals

    try:
        olympiadbench_evals.OlympiadBenchEvaluator()
    except ImportError:  # sympy's parse_latex needs antlr4 4.11; extraction does not
        monkeypatch.setattr(olympiadbench_evals, "OlympiadBenchEvaluator", lambda: None)
    name = "lmms_eval.tasks.olympiadbench_mimo.utils"
    sys.modules.pop(name, None)
    yield importlib.import_module(name)
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("First guess \\boxed{2}.\nAfter checking, the answer is \\boxed{3}.", "\\boxed{3}"),
        ("So the answers are \\boxed{1}, \\boxed{2}", "\\boxed{1}, \\boxed{2}"),
        ("Draft \\boxed{0}\nSo the answers are \\boxed{1}, \\boxed{2}.", "\\boxed{1}, \\boxed{2}"),
        ("\\boxed{\\frac{1}{2}} since $f(x)=\\frac{1}{2}$", "\\boxed{\\frac{1}{2}}"),
        ("The answer is \\boxed{\n\\frac{1}{2}\n}.", "\\boxed{\n\\frac{1}{2}\n}"),
        ("no box, the answer is 5", ""),
        ("Cut off \\boxed{3", ""),
    ],
)
def test_olympiadbench_extracts_the_final_box(mimo_utils, response, expected):
    assert mimo_utils._extract_final_boxed(response) == expected


def test_olympiadbench_scores_the_corrected_answer(mimo_utils):
    if mimo_utils.olympiadbench_evaluator is None:
        pytest.skip("OlympiadBenchEvaluator needs antlr4-python3-runtime 4.11")
    doc = {"error": None, "question_type": "Open-ended", "final_answer": ["3"], "subject": "Math", "language": "English"}
    result = mimo_utils.olympiadbench_process_results(doc, ["First guess \\boxed{2}.\nAfter checking, the answer is \\boxed{3}."])
    assert result["exact_match"] == 1

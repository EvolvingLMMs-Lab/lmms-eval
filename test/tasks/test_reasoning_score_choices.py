"""Reasoning scorer (compute_score): option-letter answers and near matches."""

import pytest

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score, extract_final_choice, relax_exact_match


def _credit(response, ground_truth):
    return compute_score("test", response, ground_truth)["acc_score"]


# --- A multiple-choice answer without a tag or a box, or in another letter case ---


@pytest.mark.parametrize(
    ("response", "ground_truth"),
    [
        ("A", "A"),
        ("(B)", "B"),
        ("B.", "B"),
        ("So the answer is B.", "B"),
        ("The parabola opens upwards, so the function is convex.\n\nA", "A"),
        ("Comparing with the choices:\n(A) 80\n(B) 60\n(C) 40\n\nThe correct value is 60.\n\nB", "B"),
        ("Arrow (c) points to the dashed line.\n\n\\boxed{c}", "C"),
        ("<answer>d</answer>", "D"),
        ("<think>Maybe \\boxed{A}? The second frame rules that out.</think>\n\nB", "B"),
        ("<think>Maybe \\boxed{A}? The second frame rules that out.</think>\n\nThe answer is B.", "B"),
    ],
)
def test_multiple_choice_answer_now_gets_credit(response, ground_truth):
    assert _credit(response, ground_truth) == 1.0


@pytest.mark.parametrize(
    ("response", "ground_truth", "expected"),
    [
        # Tagged or boxed letters, as before.
        ("<answer>B</answer>", "B", 1.0),
        ("\\boxed{B}", "B", 1.0),
        ("\\boxed{B}", "C", 0),
        ("<think>Compare the frames.</think><answer>C</answer>", "C", 1.0),
        ("<think>Compare the frames.</think><answer>C</answer>", "D", 0),
        # A box in the reasoning counts when nothing after it states a choice.
        ("So I will go with A.\n\n\\boxed{A}\n</think>\n\nThe member is a diagonal. If we add a bit more, maybe 5.75.", "A", 1.0),
        ("So the answer is \\boxed{A}.\n</think>\n\nA", "A", 1.0),
        # The option letter inside a longer boxed answer, read by parse_mcq as before.
        ("\\boxed{Step B}", "B", 1.0),
        ("\\boxed{B. a blue bike}", "B", 1.0),
        # No final choice and no tag or box: no credit, as before.
        ("The clip shows a road at night.", "B", 0),
        ("", "B", 0),
        ("Because the bike is blue, the value is 60.", "B", 0),
    ],
)
def test_multiple_choice_answers_scored_as_before(response, ground_truth, expected):
    assert _credit(response, ground_truth) == expected


def test_final_choice_follows_the_reasoning():
    assert extract_final_choice("<think>Maybe \\boxed{A}.</think>\n\nB") == "B"
    assert extract_final_choice("<think>Maybe B.</think>\\boxed{C}") == "C"
    assert extract_final_choice("\\boxed{A}\n</think>\n\nThe member is a diagonal.") == "A"
    assert extract_final_choice("The clip is dark.\n\nPerhaps the answer is D, but the bike is blue.\n\nOption C") == "C"
    assert extract_final_choice("\\boxed{Step B}") == ""
    assert extract_final_choice("The value is 60.") == ""


# --- One option is the ground truth: a list of options is not that option ---


@pytest.mark.parametrize("response", ["\\boxed{C, F}", "\\boxed{C,F}", "<answer>C and F</answer>", "\\boxed{(C) or (F)}", "\\boxed{C/F}", "\\boxed{A, C, F}"])
@pytest.mark.parametrize("ground_truth", ["C", "F"])
def test_several_options_do_not_get_credit_for_one(response, ground_truth):
    assert _credit(response, ground_truth) == 0


def test_a_repeated_option_is_still_one_option():
    assert _credit("\\boxed{C, C}", "C") == 1.0


# --- Near match of a free-form answer: symbols may differ, letters and digits may not ---


@pytest.mark.parametrize(
    ("prediction", "ground_truth", "expected"),
    [
        ("ITM Options", "DITM Options", 0.0),
        ("DITM Options", "ITM Options", 0.0),
        ("123456789", "1234567890", 0.0),
        ("1234567890", "123456789", 0.0),
        ("Joint-CNN model", "Joint-CNN model.", 1.0),
        ("(the Joint-CNN model)", "the Joint-CNN model", 1.0),
        ("$1,234,567.89", "1,234,567.89", 1.0),
        ("Joint-CNN", "joint-cnn", 1.0),
        ("Joint-CNN", "Joint-CNN", 1.0),
        ("Joint-CNN", "Joint-RNN", 0.0),
    ],
)
def test_near_match_tolerates_symbols_only(prediction, ground_truth, expected):
    assert relax_exact_match(prediction, ground_truth) == expected

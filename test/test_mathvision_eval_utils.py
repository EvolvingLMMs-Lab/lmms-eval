"""The relaxed MathVision extraction against the MATH-V reference it replaces.

Every correction is asserted in both directions: `eval_utils` keeps reproducing the
reference behaviour (so `mathvision_standard_eval` stays comparable to the paper and
the official leaderboard), and `eval_utils_relaxed` produces the corrected result.
The values below were measured against both modules, not assumed.
"""

import pytest

pytest.importorskip("latex2sympy2")

from lmms_eval.tasks.mathvision import eval_utils as ref  # noqa: E402
from lmms_eval.tasks.mathvision import eval_utils_relaxed as relaxed  # noqa: E402
from lmms_eval.tasks.mathvision.utils import mathvision_process_results  # noqa: E402
from lmms_eval.tasks.mathvision.utils_relaxed import mathvision_relaxed_process_results  # noqa: E402

# --- _fix_sqrt: a root index is well-formed already (#1506) ---


def test_indexed_root_survives_only_in_the_relaxed_path():
    # The reference braces the leading '[' of "\\sqrt[3]{8}", producing
    # "\\sqrt{[}3]{8}" - invalid LaTeX that latex2sympy cannot post-process.
    assert ref._fix_sqrt("\\sqrt[3]{8}") == "\\sqrt{[}3]{8}"
    assert relaxed._fix_sqrt("\\sqrt[3]{8}") == "\\sqrt[3]{8}"
    assert relaxed._fix_sqrt("\\sqrt[3]{2}+1") == "\\sqrt[3]{2}+1"
    assert relaxed._fix_sqrt("2\\sqrt[3]{2}") == "2\\sqrt[3]{2}"


def test_sqrt_shorthand_is_identical_in_both_paths():
    for value in ("\\sqrt{2}", "\\sqrt2", "\\sqrta"):
        assert ref._fix_sqrt(value) == relaxed._fix_sqrt(value)


# --- _strip_string: the fraction normalization is not a sqrt-only path (#1508) ---


def test_fraction_shorthand_is_normalized_only_in_the_relaxed_path():
    # "\\frac12" contains no "sqrt", so the reference never reaches _fix_fracs.
    assert ref._strip_string("\\frac12") == "\\frac12"
    assert relaxed._strip_string("\\frac12") == "\\frac{1}{2}"
    assert relaxed._strip_string("\\frac1{2}") == "\\frac{1}{2}"
    assert relaxed._strip_string("\\frac34") == "\\frac{3}{4}"
    assert relaxed._strip_string("1\\frac12") == "1\\frac{1}{2}"


def test_fraction_handling_is_identical_elsewhere():
    # "\\frac{1}2" is left alone by _fix_fracs in both paths, and a sqrt-free
    # answer containing a sqrt already normalized correctly before the fix.
    for value in ("\\frac{1}2", "\\frac12 + \\sqrt2"):
        assert ref._strip_string(value) == relaxed._strip_string(value)


# --- find_math_answer: \text / \mbox keep their content (#1510) ---


def test_text_answer_is_unwrapped_only_in_the_relaxed_path():
    # Deleting the command name but keeping the braces gives "{sunday}", which
    # is_equal scores False against the bare gold "Sunday".
    assert ref.find_math_answer("\\boxed{\\text{Sunday}}") == "{sunday}"
    assert relaxed.find_math_answer("\\boxed{\\text{Sunday}}") == "sunday"
    assert ref.is_equal("Sunday", ref.find_math_answer("\\boxed{\\text{Sunday}}")) is False
    assert ref.is_equal("Sunday", relaxed.find_math_answer("\\boxed{\\text{Sunday}}")) is True
    assert relaxed.find_math_answer("\\boxed{\\mbox{8}}") == "8"


def test_nested_text_falls_back_in_both_paths():
    # Only brace-balanced spans are unwrapped: "{\frac{1}{2}}" parses in
    # latex2sympy and must keep doing so.
    assert ref.find_math_answer("\\boxed{\\text{\\frac{1}{2}}}") == "{\\frac{1}{2}}"
    assert relaxed.find_math_answer("\\boxed{\\text{\\frac{1}{2}}}") == "{\\frac{1}{2}}"


# --- find_math_answer: the last box, not the first one the greedy regex reaches ---


def test_last_boxed_span_wins_only_in_the_relaxed_path():
    # "oxed{(.*)}" is greedy, so it spans to the last brace of the whole string
    # and the }-split fallback then resurrects the retracted first box.
    assert ref.find_math_answer("the answer is \\boxed{2}. wait, actually \\boxed{3}") == "2"
    assert relaxed.find_math_answer("the answer is \\boxed{2}. wait, actually \\boxed{3}") == "3"


def test_box_followed_by_braced_text_is_not_mangled_in_the_relaxed_path():
    # One box plus trailing LaTeX: the greedy span swallows everything up to the
    # last brace, and the brace-balanced extraction stops at the box.
    assert ref.find_math_answer("\\boxed{\\frac{1}{2}} \\text{ which is } 0.5") == "\\frac{1}{2}}{whichis"
    assert relaxed.find_math_answer("\\boxed{\\frac{1}{2}} \\text{ which is } 0.5") == "\\frac{1}{2}"


def test_box_extraction_parity_for_shapes_that_were_already_right():
    for value in ("\\boxed{3}", "\\boxed{7}", "\\boxed{\\frac{1}{2}}", "no box here"):
        assert ref.find_math_answer(value) == relaxed.find_math_answer(value)


# --- end to end through the task's process_results ---


def _doc(answer):
    return {"answer": answer, "options": [], "question": "q"}


def test_scoring_differs_only_where_a_correction_applies():
    cases = [
        ("Sunday", "\\boxed{\\text{Sunday}}"),
        ("\\frac{1}{2}", "\\boxed{\\frac12}"),
        ("8^{1/3}", "\\boxed{\\sqrt[3]{8}}"),
    ]
    for gold, prediction in cases:
        standard = mathvision_process_results(_doc(gold), [prediction])
        corrected = mathvision_relaxed_process_results(_doc(gold), [prediction])
        assert standard["mathvision_standard_eval"]["scores"] == [False]
        assert corrected["mathvision_relaxed_eval"]["scores"] == [True]


def test_scoring_agrees_on_an_unchanged_answer():
    gold, prediction = "3", "\\boxed{3}"
    standard = mathvision_process_results(_doc(gold), [prediction])
    corrected = mathvision_relaxed_process_results(_doc(gold), [prediction])
    assert standard["mathvision_standard_eval"]["scores"] == [True]
    assert corrected["mathvision_relaxed_eval"]["scores"] == [True]

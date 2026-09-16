from lmms_eval.tasks.mathvision.eval_utils import _fix_sqrt, _strip_string, find_math_answer


def test_last_boxed_answer_wins():  # #1511
    assert find_math_answer(r"\boxed{3}. Wait, actually \boxed{5}") == "5"


def test_text_wrapper_removed():  # #1509
    assert find_math_answer(r"\boxed{\text{Sunday}}") == "sunday"


def test_frac_shorthand_normalized_without_sqrt():  # #1507
    assert _strip_string(r"\frac12") == r"\frac{1}{2}"


def test_indexed_root_preserved():  # #1505
    assert _fix_sqrt(r"\sqrt[3]{8}") == r"\sqrt[3]{8}"
    assert _fix_sqrt(r"\sqrt2") == r"\sqrt{2}"


def test_nested_boxed_content():
    assert find_math_answer(r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"

import pytest

pytest.importorskip("latex2sympy2")

from lmms_eval.tasks.mathvision.eval_utils import _fix_sqrt, _strip_string  # noqa: E402


def test_fix_sqrt_preserves_indexed_roots():
    r"""\sqrt[<n>]{...} is already well-formed. Bracing its '[' corrupted it to
    '\sqrt{[}3]{8}', which is invalid LaTeX and killed the latex2sympy
    equivalence path in is_equal."""
    assert _fix_sqrt("\\sqrt[3]{8}") == "\\sqrt[3]{8}"
    assert _fix_sqrt("\\sqrt[3]{2}+1") == "\\sqrt[3]{2}+1"
    assert _fix_sqrt("2\\sqrt[3]{2}") == "2\\sqrt[3]{2}"


def test_fix_sqrt_keeps_existing_shorthand_behavior():
    assert _fix_sqrt("\\sqrt{2}") == "\\sqrt{2}"
    assert _fix_sqrt("\\sqrt2") == "\\sqrt{2}"
    assert _fix_sqrt("\\sqrta") == "\\sqrt{a}"


def test_strip_string_leaves_indexed_roots_parseable():
    from latex2sympy2 import latex2sympy

    stripped = _strip_string("\\sqrt[3]{8}")
    # the stripped gold/answer must still parse, so is_equal's sympy fallback works
    assert str(latex2sympy(stripped)) == "8**(1/3)"

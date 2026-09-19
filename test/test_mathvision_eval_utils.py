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


def test_frac_shorthand_normalized_without_sqrt():
    assert _strip_string("\\frac12") == "\\frac{1}{2}"
    assert _strip_string("\\frac1{2}") == "\\frac{1}{2}"
    assert _strip_string("\\frac34") == "\\frac{3}{4}"
    assert _strip_string("1\\frac12") == "1\\frac{1}{2}"


def test_frac_shorthand_parseable_without_sqrt():
    from latex2sympy2 import latex2sympy

    assert str(latex2sympy(_strip_string("\\frac12"))) == "1/2"


def test_frac_braced_numeral_form_is_canonical_passthrough():
    # \frac{1}2 is left as-is by the canonical _fix_fracs (hendrycks
    # math_equivalence); this documents parity, not a fix.
    assert _strip_string("\\frac{1}2") == "\\frac{1}2"


def test_frac_still_normalized_with_sqrt_present():
    # Control: normalization already worked when a sqrt was present;
    # the fix removes the guard that made sqrt presence a precondition.
    assert _strip_string("\\frac12 + \\sqrt2") == "\\frac{1}{2}+\\sqrt{2}"

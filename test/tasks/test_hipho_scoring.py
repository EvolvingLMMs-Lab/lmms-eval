"""HiPhO scoring: a response is credited only when it matches the reference answer.

The references below keep the format of the HY-Wan/HiPhO dataset: every answer is stored as
"\\boxed{...}", often with the math wrapped in "$...$" inside the box.
"""

import json

import pytest

from lmms_eval.tasks.hipho import utils


def score(refs, response, points=None):
    doc = {"id": "q", "source": "S", "field": "F", "answer": json.dumps(refs), "points": json.dumps(points if points is not None else [1.0] * len(refs))}
    return utils.hipho_process_results(doc, [response])["hipho_score"]["score"]


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Final Answer: \\boxed{A}", 1.0),
        ("Final Answer: \\boxed{B}", 0.0),
        ("Final Answer: \\boxed{C}", 0.0),
        ("Final Answer: \\boxed{D}", 0.0),
        ("Final Answer: \\boxed{E}", 0.0),
        ("Final Answer: \\boxed{\\text{A}}", 1.0),
    ],
)
def test_multiple_choice_letter_is_matched_exactly(response, expected):
    # "b", "d", "e" (and "o", "x") are letters of the "\boxed" wrapper, so they matched every reference.
    assert score(["\\boxed{A}"], response) == expected


@pytest.mark.parametrize("letter", ["x", "e", "o", "b", "d"])
def test_single_letter_does_not_match_an_expression(letter):
    assert score(["\\boxed{$\\sqrt{\\frac{\\pi}{G \\rho_0}}$}"], f"Final Answer: \\boxed{{{letter}}}") == 0.0


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Final Answer: \\boxed{\\beta = 2}", 1.0),
        ("Final Answer: \\boxed{$\\beta = 2$}", 1.0),
        ("Final Answer: \\boxed{2}", 1.0),
        ("Final Answer: \\boxed{\\beta = 3}", 0.0),
        ("Final Answer: \\boxed{\\beta = 98765}", 0.0),
        ("Final Answer: \\boxed{\\beta}", 0.0),
    ],
)
def test_left_hand_side_of_an_equation_is_not_an_answer(response, expected):
    # A wrong value for the right variable used to share the left-hand side with the reference.
    assert score(["\\boxed{$\\beta = 2$}"], response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Final Answer: \\boxed{12}", 1.0),
        ("Final Answer: \\boxed{2}", 0.0),
        ("Final Answer: \\boxed{1}", 0.0),
        ("Final Answer: \\boxed{120}", 0.0),
    ],
)
def test_fragment_of_the_reference_is_not_a_match(response, expected):
    assert score(["\\boxed{12}"], response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Final Answer: \\boxed{184.3}", 1.0),
        ("Final Answer: \\boxed{184.30 \\text{ K}}", 1.0),
        ("Final Answer: \\boxed{T = 184.3}", 1.0),
        ("Final Answer: \\boxed{190}", 0.0),
    ],
)
def test_numeric_references_keep_their_tolerance_and_units(response, expected):
    assert score(["\\boxed{184.3}"], response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Final Answer: \\boxed{\\frac{2g}{\\Omega^{2}K}}", 1.0),
        ("Final Answer: \\boxed{a = \\frac{2g}{\\Omega^{2}K}}", 1.0),
        ("Final Answer: \\boxed{\\frac{2g}{\\Omega K}}", 0.0),
        ("Final Answer: \\boxed{\\frac{g}{\\Omega^{2}K}}", 0.0),
    ],
)
def test_expression_references(response, expected):
    assert score(["\\boxed{$a = \\frac{2 g}{\\Omega^{2} K}$}"], response) == expected


def test_reference_with_leading_dollar_before_the_box():
    ref = "$\\boxed{$[5.21 \\times 10^{-1}, 5.28 \\times 10^{-1}]$}"
    assert score([ref], "Final Answer: \\boxed{[5.21 \\times 10^{-1}, 5.28 \\times 10^{-1}]}") == 1.0
    assert score([ref], "Final Answer: \\boxed{[5.21 \\times 10^{-1}, 5.30 \\times 10^{-1}]}") == 0.0


def test_partial_credit_follows_the_points_of_each_answer():
    refs = ["\\boxed{$\\beta = 2$}", "\\boxed{$\\gamma = -1$}", "\\boxed{$\\delta = 4$}"]
    points = [0.3, 0.3, 0.2]
    assert score(refs, "Final Answer: \\boxed{\\beta = 2, \\gamma = -1, \\delta = 4}", points) == pytest.approx(1.0)
    assert score(refs, "Final Answer: \\boxed{\\beta = 2, \\gamma = 5, \\delta = 9}", points) == pytest.approx(0.3 / 0.8)
    assert score(refs, "Final Answer: \\boxed{\\beta = 7, \\gamma = 5, \\delta = 9}", points) == 0.0


def test_constant_answer_scores_zero_on_a_mixed_exam():
    exam = [
        (["\\boxed{A}"], [1.0]),
        (["\\boxed{C}"], [1.0]),
        (["\\boxed{$v_1 = \\alpha c$}"], [1.0]),
        (["\\boxed{$\\beta = 2$}", "\\boxed{$\\gamma = -1$}"], [0.5, 0.5]),
    ]
    results = []
    for refs, points in exam:
        doc = {"id": "q", "source": "S", "field": "F", "answer": json.dumps(refs), "points": json.dumps(points)}
        results.append(utils.hipho_process_results(doc, ["Final Answer: \\boxed{x, x}"])["hipho_score"])
    assert utils.hipho_aggregate_results(results) == 0.0


@pytest.mark.parametrize(
    "ref, digit",
    [
        ("\\boxed{$\\gamma = \\frac{Q}{2M}$}", "2"),
        ("\\boxed{$v_e = v_0 \\exp(-\\pi \\mu)$}", "0"),
        ("\\boxed{$\\Gamma = \\frac{1}{\\tau}$}", "1"),
        ("\\boxed{$\\sqrt{\\frac{\\pi}{G \\rho_0}}$}", "0"),
    ],
)
def test_single_number_inside_an_expression_is_not_its_value(ref, digit):
    # The numeric fallback read the only number of an expression as the value of the expression.
    assert score([ref], f"Final Answer: \\boxed{{{digit}}}") == 0.0
    assert score([ref], f"Final Answer: \\boxed{{x_{digit}}}") == 0.0


@pytest.mark.parametrize(
    "response",
    [
        "Final Answer: \\boxed{\\approx 184.3}",
        "Final Answer: \\boxed{T \\approx 184.3 \\text{ K}}",
        "Final Answer: \\boxed{T \\simeq 184.3}",
        "Final Answer: \\boxed{T = 184.3 \\, \\mathrm{K}}",
    ],
)
def test_value_after_a_relation_sign_keeps_its_tolerance(response):
    assert score(["\\boxed{184.3}"], response) == 1.0
    assert score(["\\boxed{190}"], response) == 0.0

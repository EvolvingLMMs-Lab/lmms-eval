"""Contract of ``extract_final_mcq_answer``: the final answer of a response that may contain reasoning."""

import json
from pathlib import Path

import pytest

from lmms_eval.tasks._task_utils.mcq_extract import extract_final_mcq_answer, extract_mcq_answer

_CASES = json.loads((Path(__file__).with_name("mcq_extract") / "cases.json").read_text(encoding="utf-8"))["cases"]
ABCD = list("ABCD")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_never_contradicts_the_reviewed_short_answer_contract(case):
    """On the reviewed corpus the result is the reviewed answer, or an abstention; never another letter."""
    answer = extract_final_mcq_answer(case["response"], choices=case["choices"])
    assert answer in (case["expected"], "")
    if case["expected"] == "":
        assert answer == ""


@pytest.mark.parametrize(
    ("response", "expected", "rationale"),
    [
        # Rule 1: a container decides, as in extract_mcq_answer.
        ("Maybe A.\n\nThe frames show a bike.\n\n\\boxed{B}", "B", "container at the end of a reasoning chain"),
        ("<answer>C</answer>\n\nThe object is a tree, not the car in A.", "C", "container before an explanation"),
        ("First \\boxed{A}.\n\nAfter a second look \\boxed{D}", "D", "the last container wins"),
        ("The answer is B.\n\n\\boxed{Z}", "", "an invalid final container is not replaced by an earlier answer"),
        ("The format is \\boxed{}.\n\nSo the answer is C.", "", "an empty container abstains, as in extract_mcq_answer"),
        # Rule 2: several paragraphs, the first and the last one are read.
        ("The clip shows a road.\n\nPerhaps the answer is D, but the bike is blue.\n\nSo I will answer C.\n\nOption C", "C", "a letter considered in the middle is not the final answer"),
        ("Step 1: the car is red, so A looks right.\n\nStep 2: the question asks about the bike.\n\nTherefore, the correct option is B.", "B", "conclusion in the last paragraph"),
        ("The question asks for the last object.\n\nIt cannot be (A) or (B).\n\n(D)", "D", "standalone choice in the last paragraph"),
        ("B\n\nThe red car appears first, but the question asks about the last object. Option A names the car.", "B", "answer first, explanation afterwards"),
        ("The answer is C.\n\nA. a red car: appears first.\nB. a blue bike: never appears.\nD. an old book: appears second.", "C", "answer first, then a list of the other options"),
        ("**D. an old book**\n\nThe book is on the table in the last frame.", "D", "bold label first"),
        ("A. Initial guess.\n\nThe frames are dark.\n\nThe answer is C.", "", "the first and the last paragraph select different choices"),
        ("D. an old book\n\nB. a blue bike: it is not in the last frame", "", "two labelled paragraphs are ambiguous"),
        ("Answer: C\n\nThe frames are dark.\n\nFinal answer: unknown", "", "an invalid final declaration is not replaced by an earlier answer"),
        ("The answer is A.\n\nThe frames are dark.\n\nOn reflection, the answer is B.", "", "two different declarations at the two ends"),
        ("The clip is dark.\n\nI think the answer is B, but it might be C.\n\nIt is hard to tell.", "", "a declaration only in the middle is not trusted"),
        ("The clip shows a road.\n\nIt is night.", "", "no choice at either end"),
        # Rule 3: one paragraph.
        ("I see a dog. Answer: B", "B", "one line is read whole"),
        ("Answer: D\nFinal answer: a", "A", "one paragraph is read whole: the last declaration wins"),
        ("The answer is B.\n(D)", "B", "one paragraph is read whole: the declaration wins over a later label"),
        ("The two forces are T_A = 3 N and T_C = 4 N, so the bar tilts.\n**D**", "D", "prose that names letters, then a choice on the last line"),
        ("(A) and (B) are true statements; (C) is not.\n(D)", "D", "a line that lists several options selects none; the last line selects one"),
        ("There are two possibilities.\nB\nD", "", "two different standalone lines"),
        ("A. a red car\nB. a blue bike\nC. a green tree", "", "a copied option list"),
        ("Answer: C\nFinal answer: unknown", "", "an invalid declaration is not replaced by reading single lines"),
        # Inputs.
        ("", "", "empty"),
        ("  \n\n  ", "", "whitespace"),
        ("x\r\n\r\nAnswer: C", "C", "Windows line ends separate paragraphs"),
    ],
)
def test_final_answer_rules(response, expected, rationale):
    assert extract_final_mcq_answer(response, ABCD) == expected, rationale


@pytest.mark.parametrize("response", [None, 3, 2.5, {"text": "B"}, ["B"]])
def test_non_string_response_is_empty(response):
    assert extract_final_mcq_answer(response, ABCD) == ""


def test_one_line_equals_the_short_answer_contract():
    for response in ["B", "(b)", "The answer is C because it is red.", "A cat sits on a mat.", "A or B", "Vitamin B"]:
        assert extract_final_mcq_answer(response, ABCD) == extract_mcq_answer(response, ABCD)


def test_choices_restrict_the_answer():
    assert extract_final_mcq_answer("The clip is dark.\n\nAnswer: C", ["A", "B"]) == ""
    assert extract_final_mcq_answer("The clip is dark.\n\nAnswer: J", list("ABCDEFGHIJ")) == "J"
    assert extract_final_mcq_answer("The clip is dark.\n\nAnswer: c") == "C"

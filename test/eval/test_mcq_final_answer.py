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
        ("A. Initial guess.\n\nThe frames are dark.\n\nThe answer is C.", "C", "a label with text in the first paragraph states no answer; the last paragraph declares one"),
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
        # Options that are listed or weighed are not an answer.
        ("Option A: a red car. Incorrect.\n\nOption B: a blue bike. Correct.\n\nTherefore, B is correct.", "", "an analysis of option A opens the response; the conclusion is not in a format this parser reads"),
        ("A: The car is not present.\n\nHence B.", "", "a label with text in the first paragraph states no answer"),
        ("From (b) we see a green tree.\n\nSo C.", "", "a sub-figure label in prose"),
        ("B: 12 cm\n\nTherefore the length is 12 cm, which corresponds to D.", "", "a variable defined as a label"),
        ("The clip is dark.\n\nAnswer: C\n\nNote: (a) is the left panel.", "", "a sub-figure label in the last paragraph; the declaration is in the middle"),
        ("B. a blue bike\n\nThe bike is blue.", "", "a label with text and an explanation: left to the caller's own parser"),
        ("Let the triangle be ABC.\nA: 30 degrees\nB: 60 degrees", "", "labelled lines are not standalone choices"),
        # A declared letter that begins a phrase is not a choice.
        ("The answer is a blue bike.", "", "the article a"),
        ("The best option is a green tree, since it is tall.", "", "the article a"),
        ("The answer is a = 3.", "", "a variable"),
        ("The answer is I think C.", "", "the pronoun I"),
        ("Perhaps the answer is B. But only C lists all three. Therefore, C is the correct answer.", "", "a later verdict names another letter; the caller's parser decides"),
        ("The answer is C. So C is the correct answer.", "C", "a later verdict on the same letter"),
        ("The melody ends on C. The first option is C minor.\n**A**", "", "option text that starts with a letter; the declaration is unusable, so the caller's parser decides"),
        ("The answer is B (the blue bike).", "B", "a declared choice with its option text in parentheses"),
        ("The answer is A because it is red.", "A", "an assertion keeps the choice"),
        ("the answer is b because the objects match.", "B", "an assertion keeps a lowercase choice"),
        ("The answer is A.", "A", "a declared choice at the end of a sentence"),
        ("Option D is correct, as both affect the trade winds. Option B is incorrect, because the patterns are opposite.", "", "verdicts on several options"),
        ("The frames are dark.\n\nOption D is correct, as both affect the trade winds. Option B is incorrect.", "", "verdicts on several options in the last paragraph"),
        ("The frames are dark.\n\nOption D is correct.", "D", "one verdict selects the option"),
        ("Let me check each option: (A) shows a curve on both sides. So (C) is correct.", "", "an option list opened with a colon"),
        ("Option: B", "B", "a declared option that stands alone"),
        ("Answer: (B) a blue bike", "B", "a declared answer with its option text"),
        # A letter met inside prose is not read.
        ("Based on subfigure (b), the tree is green.", "", "a sub-figure label"),
        ("Now compute C = 50 * 0.58 - 21.9, which rounds to 7.34.", "", "a variable"),
        ("The red object is above the table.\nB", "B", "a standalone choice on the last line"),
        ("(B) a blue bike", "B", "the paragraph opens with the choice"),
        ("B is correct", "B", "the paragraph opens with the choice"),
        # A container holds a choice, not an expression.
        ("\\boxed{b^2 + 1}", "", "an expression"),
        ("\\boxed{A\\cup B}", "", "a set expression"),
        ("\\boxed{a > b}", "", "an inequality"),
        ("\\boxed{B. a blue bike}", "B", "a label with its option text"),
        ("\\boxed{(B) a blue bike}", "B", "a parenthesized label with its option text"),
        ("\\boxed{\\text{B}}", "B", "a presentation wrapper"),
        ("\\boxed{Answer: B}", "B", "a declaration inside the container"),
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

"""Option-letter scoring for SPAR-Bench and ScienceQA: formats that scored wrong before."""

import pytest

from lmms_eval.tasks.scienceqa.utils import sqa_process_results
from lmms_eval.tasks.sparbench.utils import exact_match


@pytest.mark.parametrize(
    ("pred", "target", "expected"),
    [
        ("", "B", 0),  # an empty response is a substring of every target
        ("   ", "B", 0),
        ("1", "12", 0),
        ("B", "B", 1.0),
        ("b", "B", 1.0),
        ("B. the left", "B", 1.0),
        ("(B)", "B", 1.0),
        ("The answer is B", "B", 1.0),
        ("Answer: B", "B", 1.0),
        ("A", "B", 0),
        ("A or B", "B", 0),
        ("Because the object is on the left.", "B", 0),
    ],
)
def test_sparbench_exact_match(pred, target, expected):
    assert exact_match(pred, target) == expected


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        ("A", 1.0),
        ("a", 1.0),
        ("A. x", 1.0),
        ("(A)", 1.0),
        ("Answer: A", 1.0),
        ("The answer is A.", 1.0),
        ("**A**", 1.0),
        ("B", 0.0),
        ("B. y", 0.0),
        ("", 0.0),
        ("A or B", 0.0),
        ("C", 0.0),  # not an offered option
    ],
)
def test_scienceqa_process_results(pred, expected):
    doc = {"choices": ["x", "y"], "answer": 0}
    assert sqa_process_results(doc, [pred]) == {"exact_match": expected}


# --- Legacy option-letter parsers: a declared answer wins, the task's own parser stays the fallback ---

DECLARED_B = [
    "B",
    "(B)",
    "B. a blue bike",
    "The answer is B.",
    "Answer: B",
    "**B**",
    "\\boxed{B}",
    "<answer>B</answer>",
    "I see a dog. Answer: B",
    "Considering the clip, the answer is B",
    "A dog runs, so the answer is B.",
]
INDEX2ANS = {"A": "a red car", "B": "a blue bike", "C": "a tree", "D": "a book"}


def _mmmu(response):
    from lmms_eval.tasks._task_utils.mmmu_mcq_utils import parse_mmmu_multi_choice_response

    return parse_mmmu_multi_choice_response(response, list("ABCD"), INDEX2ANS)


def _jmmmu(response):
    from lmms_eval.tasks._task_utils.mmmu_mcq_utils import parse_jmmmu_multi_choice_response

    return parse_jmmmu_multi_choice_response(response, list("ABCD"), INDEX2ANS)


def _longvideobench(response):
    from lmms_eval.tasks.longvideobench.utils import parse_multi_choice_response

    return parse_multi_choice_response(response, list("ABCDE"), {})


def _egoschema(response):
    from lmms_eval.tasks.egoschema.utils import parse_multi_choice_response

    return parse_multi_choice_response(response, list("ABCDE"), INDEX2ANS)[0]


def _mmvp(response):
    from lmms_eval.tasks.mmvp.utils import _extract_answer_letter

    return _extract_answer_letter(response)


def _vsibench(response):
    from lmms_eval.tasks.vsibench.utils import fuzzy_matching

    return fuzzy_matching(response)


def _mvbench(response):
    from lmms_eval.tasks.mvbench.utils import mcq_acc

    return "B" if mcq_acc("B", response) else ""


@pytest.mark.parametrize("parse", [_mmmu, _jmmmu, _longvideobench, _egoschema, _mmvp, _vsibench, _mvbench])
@pytest.mark.parametrize("response", DECLARED_B)
def test_legacy_parsers_read_a_declared_answer(parse, response):
    assert parse(response) == "B"


def test_legacy_parsers_keep_their_own_fallback():
    # The shared extractor abstains here, so each task's parser decides as before.
    assert _longvideobench("it moves left, which D shows") == "D"
    assert _mmmu("It shows a blue bike on the road near the shop.") == "B"
    assert _vsibench("left") == "left"
    assert _mmvp("") == ""

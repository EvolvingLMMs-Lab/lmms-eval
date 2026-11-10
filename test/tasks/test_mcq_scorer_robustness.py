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

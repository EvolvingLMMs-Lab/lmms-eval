import pytest

from lmms_eval.tasks.geometry3k import utils

CHOICES = ["30", "45", "60", "90"]


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Option B is a better fit.", "B"),
        ("B, since it is a right angle", "B"),
        ("Choice B gives a perimeter of 12", "B"),
        ("C", "C"),
        ("c", "C"),
        ("Answer: d", "D"),
        ("The answer is C.", "C"),
        ("C) 60", "C"),
    ],
)
def test_extract_answer_letter(response, expected):
    assert utils._extract_answer_letter(response, CHOICES) == expected


def test_process_results_does_not_read_the_article_a_as_a_choice():
    doc = {"ground_truth": "B", "choices": CHOICES}
    assert utils.geometry3k_process_results(doc, ["Option B is a better fit."]) == {"exact_match": 1.0}

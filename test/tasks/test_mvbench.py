import pytest

from lmms_eval.tasks.mvbench.utils import mcq_acc


@pytest.mark.parametrize(
    ("pred", "gt", "expected_score"),
    [
        # Bare option letters (case-insensitive)
        ("A", "A", 1),
        ("b", "B", 1),
        ("C", "c", 1),
        ("d", "D", 1),
        ("E", "e", 1),
        ("B", "A", 0),
        # Formatted option letters
        ("A. running", "A", 1),
        ("B. jumping", "B", 1),
        ("(C)", "C", 1),
        ("The answer is B.", "B", 1),
        # Truncated or prose response containing the English article "a" (#1694)
        (
            "Looking at the sequence of images, I can see a person's hand appearing in",
            "B",
            0,
        ),
        (
            "Looking at the sequence of images, I can see a person's hand appearing in",
            "A",
            0,
        ),
        # Sentence that starts with the article "A" (capital) is not option A
        ("A person walks into the room", "A", 0),
        ("A cat sits on a table", "A", 0),
        ("a person", "A", 0),
        # Uppercase B-E in prose is still read
        ("I can see a B", "B", 1),
        # Sentence containing article "a" before the actual answer letter
        (
            "I see a cat in the video, so the answer is B.",
            "B",
            1,
        ),
    ],
)
def test_mcq_acc(pred, gt, expected_score):
    assert mcq_acc(gt, pred) == expected_score

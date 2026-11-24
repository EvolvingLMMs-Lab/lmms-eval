"""CharXiv accuracy counts failed or invalid gradings as wrong, as the official get_stats.py does."""

import pytest

from lmms_eval.tasks.charxiv import utils
from lmms_eval.tasks.charxiv.descriptive_utils import build_dummy_output


def _reasoning_results(n):
    docs = [
        {
            "figure_path": f"images/{i}.jpg",
            "reasoning_q": "How many data points are above 50?",
            "reasoning_a": "3",
            "reasoning_a_type": 4,
        }
        for i in range(n)
    ]
    return [utils.charxiv_reasoning_process_results(doc, ["3"])["reasoning_acc"] for doc in docs]


def _descriptive_results(n):
    return [{"group": (f"images/{i}.jpg_0", "2", "2"), "qid": 1} for i in range(n)]


@pytest.fixture(autouse=True)
def no_client(monkeypatch):
    monkeypatch.setattr(utils, "_get_client", lambda: None)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([1, 1, 1, 1], 1.0),
        ([1, 0, 1, -1], 0.5),
        ([-1, -1, -1, -1], 0.0),
        ([1, "1", 2, 0], 0.25),
    ],
    ids=["all-correct", "one-failed", "all-failed", "non-binary"],
)
def test_reasoning_accuracy_counts_invalid_scores_as_zero(monkeypatch, scores, expected):
    replies = iter(scores)
    monkeypatch.setattr(utils, "get_reasoning_result_gpt", lambda client, prompt, model: ("3", next(replies)))

    assert utils.charxiv_reasoning_aggregate_results(_reasoning_results(len(scores))) == pytest.approx(expected)


def test_descriptive_accuracy_counts_failed_batches_as_zero(monkeypatch):
    # Six responses for one template make two grading batches (5 + 1); the first batch fails.
    replies = iter([build_dummy_output(5), {"extract_answer_T1": "2", "score_T1": 1}])
    monkeypatch.setattr(utils, "get_descriptive_result_gpt", lambda client, prompt, length, model: next(replies))

    assert utils.charxiv_descriptive_aggregate_results(_descriptive_results(6)) == pytest.approx(1 / 6)

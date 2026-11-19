"""MMVU open-ended scoring: the rule layer credits only replies that really contain the reference.

Before this check, an empty reply was a "substring" of every reference and a one-letter reply was a
substring of most of them, so both were scored correct and never reached the GPT judge.
"""

from types import SimpleNamespace

import pytest

from lmms_eval.tasks.mmvu import utils


def open_ended(answer):
    return {"id": "q1", "question_type": "open-ended", "question": "Which one?", "answer": answer, "video_path": "videos/Computer_Science/q1.mp4"}


def rule_based(answer, reply):
    return utils.evaluate_with_rule_based(open_ended(answer), reply)


class StubJudge:
    """Stands in for the GPT judge and records what it was asked to grade."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.predictions = []

    def __call__(self, question, prediction, ground_truth):
        self.predictions.append(prediction)
        return SimpleNamespace(is_correct=self.verdict)


@pytest.fixture
def judge(monkeypatch):
    stub = StubJudge(verdict=False)
    monkeypatch.setattr(utils, "_get_pipeline", lambda: stub)
    return stub


@pytest.mark.parametrize("reply", ["", " ", "\n\t "])
@pytest.mark.parametrize("answer", ["Depth-First Search (DFS)", "capacitor", "2.4", "n + k"])
def test_empty_reply_is_not_accepted_by_the_rule_layer(answer, reply):
    assert rule_based(answer, reply) is False


@pytest.mark.parametrize(
    "answer, reply",
    [
        ("Depth-First Search (DFS)", "a"),
        ("Depth-First Search (DFS)", "-"),
        ("Depth-First Search (DFS)", "first"),
        ("High-pass filter", "a"),
        ("High-pass filter", "filter"),
        ("capacitor", "cap"),
        ("capacitor", "t"),
        ("lunar eclipse", "e"),
        ("Gastro-Oesophageal Reflux Disease (GORD)", "gord"),
    ],
)
def test_fragment_of_the_reference_is_not_accepted_by_the_rule_layer(answer, reply):
    assert rule_based(answer, reply) is False


@pytest.mark.parametrize("reply", ["qwerty nonsense", "a", "k = n"])
def test_answer_made_only_of_short_words_is_not_matched_vacuously(reply):
    # Every word of "n + k" is 2 characters or fewer, so the word-level check had nothing to compare.
    assert rule_based("n + k", reply) is False


@pytest.mark.parametrize(
    "answer, reply",
    [
        ("Depth-First Search (DFS)", "Depth-First Search (DFS)"),
        ("Depth-First Search (DFS)", "  depth-first   SEARCH (dfs) "),
        ("Depth-First Search (DFS)", "The algorithm is depth-first search (DFS)."),
        ("capacitor", "It is a capacitor."),
        ("2.4 m/s", "2.40 m/s"),
        ("lunar eclipse", "an eclipse of the lunar kind"),
        ("O(n²)", "The complexity is O(n^2)"),
        ("n + k", "n + k"),
    ],
)
def test_correct_replies_are_still_accepted(answer, reply):
    assert rule_based(answer, reply) is True


def test_empty_reply_is_left_to_the_judge(judge):
    correct, method = utils.evaluate_with_llm_judge(open_ended("Depth-First Search (DFS)"), "")
    assert (correct, method) == (False, "gpt-based")
    assert judge.predictions == [""]


def test_fragment_is_left_to_the_judge_which_can_still_accept_it(judge):
    judge.verdict = True
    correct, method = utils.evaluate_with_llm_judge(open_ended("Gastro-Oesophageal Reflux Disease (GORD)"), "GORD")
    assert (correct, method) == (True, "gpt-based")
    assert judge.predictions == ["GORD"]


def test_exact_answer_is_accepted_without_calling_the_judge(judge):
    correct, method = utils.evaluate_with_llm_judge(open_ended("Depth-First Search (DFS)"), "Depth-First Search (DFS)")
    assert (correct, method) == (True, "rule-based")
    assert judge.predictions == []


def test_process_results_scores_an_empty_reply_as_incorrect(judge):
    result = utils.mmvu_process_results(open_ended("capacitor"), [""])["accuracy"]
    assert result["correct"] == 0
    assert result["eval_method"] == "gpt-based"

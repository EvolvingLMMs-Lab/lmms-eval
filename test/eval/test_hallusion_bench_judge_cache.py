import os
from types import SimpleNamespace

import pytest

from lmms_eval.tasks.hallusion_bench import evaluate_hb as hb
from lmms_eval.tasks.hallusion_bench import utils as hb_utils

ANSWERS = ["yes", "no", "yes", "no"]


class _Judge:
    """Stands in for the GPT judge: 'correct' iff the prediction starts with the same Yes/No as the reference."""

    def __init__(self):
        self.calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        prompt = json["messages"][0]["content"]
        reference = prompt.split("Reference answer: ")[1].split("\nPrediction answer:")[0]
        prediction = prompt.split("Prediction answer:")[1].split("\nOutput:")[0]
        verdict = "correct" if reference.split()[0].strip(",.").lower() == prediction.split()[0].strip(",.").lower() else "incorrect"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"choices": [{"message": {"content": verdict}}]})


@pytest.fixture
def judge(monkeypatch):
    judge = _Judge()
    monkeypatch.setattr(hb_utils.requests, "post", judge.post)
    return judge


def _results(predictions):
    return [
        {
            "category": "VD",
            "subcategory": "illusion",
            "visual_input": "1",
            "set_id": "0",
            "figure_id": str(i),
            "question_id": "0",
            "question": f"Is it {i}?",
            "gt_answer": "1" if answer == "yes" else "0",
            "gt_answer_details": f"{answer.capitalize()}, it is.",
            "model_prediction": prediction,
        }
        for i, (answer, prediction) in enumerate(zip(ANSWERS, predictions))
    ]


def _right():
    return [answer.capitalize() for answer in ANSWERS]


def _wrong():
    return ["No" if answer == "yes" else "Yes" for answer in ANSWERS]


def test_second_model_does_not_reuse_judgments_of_the_first_in_the_same_output_path(judge, tmp_path):
    args = SimpleNamespace(output_path=str(tmp_path))

    assert hb.hb_aggregation_result_aAcc(_results(_right()), args) == 100.0
    assert hb.hb_aggregation_result_aAcc(_results(_wrong()), args) == 0.0
    assert judge.calls == 2 * len(ANSWERS)


def test_metrics_of_one_run_share_the_judgments(judge, tmp_path):
    args = SimpleNamespace(output_path=str(tmp_path))
    results = _results(_right())

    assert hb.hb_aggregation_result_aAcc(results, args) == 100.0
    assert hb.hb_aggregation_result_qAcc(results, args) == 100.0
    assert hb.hb_aggregation_result_fAcc(results, args) == 100.0
    assert judge.calls == len(ANSWERS)


@pytest.mark.parametrize("args", [SimpleNamespace(output_path=None), None])
def test_aggregation_without_output_path(judge, tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)

    assert hb.hb_aggregation_result_aAcc(_results(_right()), args) == 100.0
    assert os.listdir(tmp_path / "gpt_response")

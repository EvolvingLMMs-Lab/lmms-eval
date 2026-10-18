"""Tests for POPE yes-ratio aggregation without data or model downloads."""

import pytest

from lmms_eval.tasks.pope import utils


@pytest.mark.parametrize(
    ("answers", "predictions", "expected"),
    [
        pytest.param(["yes", "no"], ["yes", "yes"], 1.0, id="all-yes"),
        pytest.param(["yes", "no"], ["no", "no"], 0.0, id="all-no"),
        pytest.param(["yes", "yes", "yes", "no"], ["yes", "no", "no", "no"], 0.25, id="unbalanced-labels"),
        pytest.param(["yes", "yes"], ["yes", "no"], 0.5, id="positive-labels"),
        pytest.param(["no", "no"], ["yes", "no"], 0.5, id="negative-labels"),
        pytest.param(["yes", "no"], [" YES ", "No"], 0.5, id="normalized-predictions"),
        pytest.param(["yes", "yes", "no"], ["yes", "unsure", ""], 1 / 3, id="all-responses-in-denominator"),
        pytest.param(["yes", "no"], ["unsure", ""], 0.0, id="no-yes-or-no-predictions"),
        pytest.param([], [], 0.0, id="empty-results"),
    ],
)
def test_pope_yes_ratio_counts_predictions(answers: list[str], predictions: list[str], expected: float) -> None:
    results = [utils.pope_process_results({"question_id": index, "answer": answer}, [prediction])["pope_yes_ratio"] for index, (answer, prediction) in enumerate(zip(answers, predictions))]

    assert utils.pope_aggregate_yes_ratio(results) == pytest.approx(expected)

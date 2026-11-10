"""Tests for the --baseline paired comparison (lmms_eval/baselines/comparison.py)."""

import json
import math

import pytest

from lmms_eval.baselines.comparison import (
    compare_with_baseline,
    paired_key,
    sample_score,
    scores_by_doc,
)
from lmms_eval.baselines.loader import load_baseline
from lmms_eval.utils import make_table

# gsm8k-shaped rows: the target is the full solution text and the filters
# extract a number, so a string match of target against the response never holds.
_STRICT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_FLEXIBLE = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def _gsm8k_samples(strict=_STRICT, flexible=_FLEXIBLE, with_filter=True):
    rows = []
    for filter_key, scores in (("strict-match", strict), ("flexible-extract", flexible)):
        for doc_id, score in enumerate(scores):
            row = {
                "doc_id": doc_id,
                "target": f"Some working.\n#### {doc_id + 10}",
                "filtered_resps": [str(doc_id + 10)],
                "exact_match": score,
            }
            if with_filter:
                row["filter"] = filter_key
            rows.append(row)
    return rows


def _gsm8k_results():
    return {
        "alias": "gsm8k",
        "exact_match,strict-match": sum(_STRICT) / len(_STRICT),
        "exact_match_stderr,strict-match": 0.0,
        "exact_match,flexible-extract": sum(_FLEXIBLE) / len(_FLEXIBLE),
        "exact_match_stderr,flexible-extract": 0.2,
    }


def test_a_run_compared_with_its_own_samples_shows_no_difference():
    samples = _gsm8k_samples()

    comparison = compare_with_baseline(_gsm8k_results(), samples, samples)

    for filter_key in ("strict-match", "flexible-extract"):
        assert comparison[paired_key("exact_match", "mean_diff", filter_key)] == 0.0
        assert comparison[paired_key("exact_match", "p_value", filter_key)] == 1.0
        assert comparison[paired_key("exact_match", "n", filter_key)] == 6


def test_each_filter_is_paired_with_the_same_filter_of_the_baseline():
    current = _gsm8k_samples(strict=[1.0] * 6)
    baseline = _gsm8k_samples()

    comparison = compare_with_baseline(_gsm8k_results(), current, baseline)

    assert comparison[paired_key("exact_match", "mean_diff", "strict-match")] == 1.0
    assert comparison[paired_key("exact_match", "baseline_mean", "strict-match")] == 0.0
    assert comparison[paired_key("exact_match", "mean_diff", "flexible-extract")] == 0.0
    assert comparison[paired_key("exact_match", "baseline_mean", "flexible-extract")] == pytest.approx(4 / 6)


def test_a_dict_metric_is_read_through_score_key():
    results = {"videomme_perception_score,none": 50.0}
    current = [{"doc_id": i, "filter": "none", "videomme_perception_score": {"pred_answer": "A", "answer": "A", "score": s}} for i, s in enumerate([1.0, 0.0, 1.0, 1.0])]
    baseline = [{"doc_id": i, "filter": "none", "videomme_perception_score": {"pred_answer": "A", "answer": "B", "score": s}} for i, s in enumerate([0.0, 0.0, 1.0, 0.0])]

    comparison = compare_with_baseline(results, current, baseline)

    assert comparison[paired_key("videomme_perception_score", "mean_diff", "none")] == 0.5
    assert comparison[paired_key("videomme_perception_score", "n", "none")] == 4


def test_a_baseline_file_without_filters_pairs_when_each_doc_appears_once():
    current = [{"doc_id": i, "filter": "none", "acc": s} for i, s in enumerate([1.0, 1.0, 0.0])]
    baseline = [{"doc_id": i, "acc": s} for i, s in enumerate([1.0, 0.0, 0.0])]

    comparison = compare_with_baseline({"acc,none": 2 / 3}, current, baseline)

    assert comparison[paired_key("acc", "mean_diff", "none")] == pytest.approx(1 / 3)
    assert comparison[paired_key("acc", "n", "none")] == 3


def test_a_baseline_file_without_filters_is_refused_when_docs_repeat():
    current = _gsm8k_samples()
    baseline = _gsm8k_samples(with_filter=False)

    with pytest.raises(ValueError, match="can't be paired"):
        compare_with_baseline(_gsm8k_results(), current, baseline)


def test_metrics_without_a_per_sample_number_are_not_compared():
    results = {"mmmu_acc,none": 0.5, "submission,none": None}
    samples = [{"doc_id": 0, "filter": "none", "mmmu_acc": {"id": "x", "parsed_pred": "A", "answer": "A"}, "submission": {"id": "x"}}]

    assert compare_with_baseline(results, samples, samples) == {}


def test_a_single_shared_doc_is_not_compared():
    current = [{"doc_id": i, "filter": "none", "acc": s} for i, s in enumerate([1.0, 0.0])]
    baseline = [{"doc_id": 0, "filter": "none", "acc": 0.0}]

    assert compare_with_baseline({"acc,none": 0.5}, current, baseline) == {}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "0.5", [1.0]])
def test_sample_score_skips_values_that_are_not_a_finite_number(value):
    assert sample_score({"acc": value}, "acc") is None


def test_sample_score_counts_booleans():
    assert sample_score({"acc": True}, "acc") == 1.0
    assert sample_score({"acc": {"score": False}}, "acc") == 0.0


def test_scores_by_doc_reads_only_the_requested_filter():
    scores = scores_by_doc(_gsm8k_samples(), "exact_match", "flexible-extract")

    assert scores == dict(enumerate(_FLEXIBLE))


def test_load_baseline_returns_the_sample_rows(tmp_path):
    path = tmp_path / "20260101_000000_samples_gsm8k.jsonl"
    rows = _gsm8k_samples()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "\n")

    samples, agg = load_baseline(str(path), "gsm8k")

    assert samples == rows
    assert agg is None


def test_the_table_shows_each_filters_own_comparison():
    current = _gsm8k_samples(strict=[1.0] * 6)
    task_results = _gsm8k_results()
    task_results["paired_baseline"] = "run_a"
    task_results.update(compare_with_baseline(task_results, current, _gsm8k_samples()))
    result_dict = {"results": {"gsm8k": task_results}, "n-shot": {"gsm8k": 0}, "higher_is_better": {"gsm8k": {"exact_match": True}}}

    table = make_table(result_dict)

    lines = table.splitlines()
    headers = [cell.strip() for cell in lines[0].split("|")]
    rows = {cells[2]: cells for cells in ([cell.strip() for cell in line.split("|")] for line in lines[2:]) if len(cells) == len(headers)}
    diff = headers.index("Diff")
    assert rows["strict-match"][diff] == "+1.0000"
    assert rows["flexible-extract"][diff] == "+0.0000"
    assert rows["flexible-extract"][headers.index("P_Value")] == "1.0000"
    assert "paired" not in table
    assert not any(math.isnan(v) for v in task_results.values() if isinstance(v, float))

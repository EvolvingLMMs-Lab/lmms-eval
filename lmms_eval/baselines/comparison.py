"""Paired comparison of a run against a baseline samples file.

Both sides go through ``sample_score``, and samples are paired by
``(doc_id, filter)``, so a task with several filters gets one comparison per
metric and filter instead of mixing them.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lmms_eval.api.metrics import paired_ttest

_NON_METRIC_SUFFIXES = (
    "_stderr",
    "_stderr_clt",
    "_stderr_clustered",
    "_expected_accuracy",
    "_consensus_accuracy",
    "_internal_variance",
    "_consistency_rate",
)

PAIRED_FIELDS = ("baseline_mean", "mean_diff", "ci_lower", "ci_upper", "p_value", "n")


def paired_key(metric: str, field: str, filter_key: str) -> str:
    return f"{metric}_paired_{field},{filter_key}"


def is_paired_metric(metric: str) -> bool:
    return "_paired_" in metric


def sample_score(sample: Dict[str, Any], metric: str, score_key: str = "score") -> Optional[float]:
    """Return the per-sample value of ``metric``, or None if it has no single number."""
    value = sample.get(metric)
    if isinstance(value, dict):
        value = value.get(score_key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def task_metrics(task_results: Dict[str, Any]) -> List[Tuple[str, str]]:
    """List the (metric, filter) pairs a task reported."""
    pairs = []
    for key in task_results:
        metric, sep, filter_key = key.partition(",")
        if not sep or metric.endswith(_NON_METRIC_SUFFIXES) or is_paired_metric(metric):
            continue
        pairs.append((metric, filter_key))
    return pairs


def scores_by_doc(
    samples: Iterable[Dict[str, Any]],
    metric: str,
    filter_key: str,
    score_key: str = "score",
) -> Dict[Any, float]:
    """Map doc_id to the sample's score for one metric and filter.

    Rows without a ``filter`` field (samples files written before it was
    logged) are accepted only when no doc_id repeats, since a repeat means the
    rows of several filters can't be told apart.
    """
    rows = [s for s in samples if s.get("filter", filter_key) == filter_key]
    doc_ids = [s.get("doc_id") for s in rows]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"samples repeat doc_ids for filter '{filter_key}', so they can't be paired; rerun with --log_samples on a version that logs the filter of each sample")
    scores = {}
    for sample in rows:
        score = sample_score(sample, metric, score_key)
        if score is not None:
            scores[sample.get("doc_id")] = score
    return scores


def compare_with_baseline(
    task_results: Dict[str, Any],
    current_samples: List[Dict[str, Any]],
    baseline_samples: List[Dict[str, Any]],
    score_key: str = "score",
) -> Dict[str, Any]:
    """Run one paired t-test per reported metric and filter.

    Returns the ``<metric>_paired_<field>,<filter>`` entries to add to the
    task's results. A metric with no per-sample number on both sides is left
    out.
    """
    out: Dict[str, Any] = {}
    for metric, filter_key in task_metrics(task_results):
        current = scores_by_doc(current_samples, metric, filter_key, score_key)
        if not current:
            continue
        baseline = scores_by_doc(baseline_samples, metric, filter_key, score_key)
        doc_ids = [doc_id for doc_id in current if doc_id in baseline]
        if not doc_ids:
            continue
        current_scores = [current[doc_id] for doc_id in doc_ids]
        baseline_scores = [baseline[doc_id] for doc_id in doc_ids]
        result = paired_ttest(current_scores, baseline_scores)
        result["baseline_mean"] = sum(baseline_scores) / len(baseline_scores)
        for field in PAIRED_FIELDS:
            out[paired_key(metric, field, filter_key)] = result[field]
    return out

"""Group weights must come from each metric's own count, not from whichever metric was seen last."""

import pytest

from lmms_eval.api.group import ConfigurableGroup
from lmms_eval.api.metrics import mean, mean_stderr, pooled_sample_stderr
from lmms_eval.evaluator_utils import TaskOutput, consolidate_group_results, consolidate_results

from .test_group_postprocess import _NamedTask

ANSWER = {"exact_match": 1.0}
PROOF = {"submission": 1.0}
# Like olympiadbench_mimo: proof problems produce "submission", the rest "exact_match".
ANSWER_FIRST = [ANSWER, PROOF, ANSWER, ANSWER]
PROOF_FIRST = [PROOF, ANSWER, ANSWER, ANSWER]
UNIFORM = [{"exact_match": score, "submission": 0.0} for score in [1.0, 0.0, 0.0, 0.0]]


class _MeanTask(_NamedTask):
    def aggregation(self):
        return {"exact_match": mean, "submission": mean}

    def higher_is_better(self):
        return {"exact_match": True, "submission": True}


def _task_output(task_name, docs):
    output = TaskOutput(task=_MeanTask(task_name), task_name=task_name, task_config={})
    for doc_metrics in docs:
        for metric, value in doc_metrics.items():
            output.sample_metrics[(metric, "none")].append(value)
    output.sample_len = len(docs)
    output.calculate_aggregate_metric(bootstrap_iters=100)
    return output


def _group(metric):
    return ConfigurableGroup(
        config={
            "group": f"group_{metric}",
            "task": ["mixed", "uniform"],
            "aggregate_metric_list": [{"metric": metric, "aggregation": "mean", "filter_list": ["none"], "weight_by_size": True}],
            "metadata": {"version": 1},
        }
    )


@pytest.mark.parametrize("mixed_docs", [ANSWER_FIRST, PROOF_FIRST], ids=["answer_first", "proof_first"])
def test_weight_by_size_group_uses_each_metrics_count(mixed_docs):
    eval_tasks = [_task_output("mixed", mixed_docs), _task_output("uniform", UNIFORM)]
    assert eval_tasks[0].sample_counts == {"exact_match,none": 3, "submission,none": 1}

    results, _, _, versions, _, _ = consolidate_results(eval_tasks)
    task_dict = {_group(metric): {name: _MeanTask(name) for name in ["mixed", "uniform"]} for metric in ["exact_match", "submission"]}
    results, _, _, _ = consolidate_group_results(results, versions, task_dict)

    exact_match = results["group_exact_match"]
    assert exact_match["exact_match,none"] == pytest.approx((3 * 1.0 + 4 * 0.25) / 7)
    assert exact_match["exact_match_stderr,none"] == pytest.approx(pooled_sample_stderr([mean_stderr([1.0] * 3), mean_stderr([1.0, 0.0, 0.0, 0.0])], [3, 4]))
    assert exact_match["sample_counts"] == {"exact_match,none": 7}
    assert exact_match["samples"] == 8

    submission = results["group_submission"]
    assert submission["submission,none"] == pytest.approx((1 * 1.0 + 4 * 0.0) / 5)
    assert submission["submission_stderr,none"] == "N/A"  # a single submission item has no stderr
    assert submission["sample_counts"] == {"submission,none": 5}
    assert submission["samples"] == 8

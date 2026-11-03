"""Tests for the opt-in group postprocess_results hook."""

import os
import tempfile
import textwrap

from lmms_eval.api.group import ConfigurableGroup
from lmms_eval.api.task import Task
from lmms_eval.evaluator_utils import _resolve_group_postprocess_callable, consolidate_group_results
from lmms_eval.tasks.sitebench.utils import sitebench_merge_results
from lmms_eval.utils import load_yaml_config

AGGREGATE_METRIC_LIST = [{"metric": "acc", "aggregation": "mean", "filter_list": ["none"], "weight_by_size": True}]


class _NamedTask(Task):
    """Task subclass with the attributes consolidate_group_results reads."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.VERSION = "1.0"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc_id, ctx, **kwargs):
        return []

    def process_results(self, doc, results):
        return {}

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}


def marker_hook(group_name, subtasks, results, **kwargs):
    return {"marker,none": 1}


def _yaml_hook(group_name, subtasks, results, **kwargs):
    return {"from_yaml,none": 3}


def _task_dict(group, *names):
    return {group: {name: _NamedTask(name) for name in names}}


def _weighted_results():
    return {
        "t1": {"alias": "t1", "acc,none": 0.4, "acc_stderr,none": 0.0, "samples": 2},
        "t2": {"alias": "t2", "acc,none": 0.8, "acc_stderr,none": 0.0, "samples": 8},
        "nohook": {"alias": "nohook"},
    }


def _consolidate(group, results, **kwargs):
    return consolidate_group_results(results, {}, _task_dict(group, "t1", "t2"), **kwargs)


def test_unset_hook_keeps_default_group_metrics(monkeypatch):
    """Groups without a hook keep the pre-hook aggregation, including explicit null."""
    resolved = []
    monkeypatch.setattr(
        "lmms_eval.evaluator_utils._resolve_group_postprocess_callable",
        lambda raw: resolved.append(raw),
    )

    configs = [
        {
            "group": "nohook",
            "task": ["t1", "t2"],
            "aggregate_metric_list": AGGREGATE_METRIC_LIST,
            "metadata": {"version": 1},
        },
        {
            "group": "nohook",
            "task": ["t1", "t2"],
            "aggregate_metric_list": AGGREGATE_METRIC_LIST,
            "metadata": {"version": 1},
            "postprocess_results": None,
        },
    ]
    outputs = []
    for config in configs:
        group = ConfigurableGroup(config=config)
        assert "postprocess_results" not in group.config
        results_out, versions_out, show_group_table, _ = _consolidate(group, _weighted_results())
        outputs.append((results_out, versions_out, show_group_table))

    assert outputs[0] == outputs[1]
    results_out, versions_out, show_group_table = outputs[0]
    group_metrics = results_out["nohook"]
    assert group_metrics["acc,none"] == 0.72
    assert group_metrics["samples"] == 10
    assert group_metrics["acc_stderr,none"] == 0.0
    assert group_metrics["alias"] == "nohook"
    assert " " not in group_metrics
    assert "marker,none" not in group_metrics
    assert set(group_metrics) == {"alias", "acc,none", "acc_stderr,none", "samples"}
    assert show_group_table is True
    assert versions_out["nohook"] == 1
    assert resolved == []


def test_group_without_aggregation_stays_placeholder():
    group = ConfigurableGroup(config={"group": "plain", "task": ["t1", "t2"]})
    results = {
        "t1": {"alias": "t1", "acc,none": 0.4, "samples": 2},
        "t2": {"alias": "t2", "acc,none": 0.8, "samples": 8},
        "plain": {"alias": "plain"},
    }
    results_out, versions_out, show_group_table, _ = _consolidate(group, results)
    assert results_out["plain"] == {"alias": "plain", " ": " "}
    assert show_group_table is False
    assert "plain" not in versions_out


def test_postprocess_hook_merges_without_replacing_default_metric():
    def hook(group_name, subtasks, results, **kwargs):
        assert group_name == "nohook"
        assert set(subtasks) == {"t1", "t2"}
        return {"marker,none": 5}

    group = ConfigurableGroup(
        config={
            "group": "nohook",
            "task": ["t1", "t2"],
            "aggregate_metric_list": AGGREGATE_METRIC_LIST,
            "postprocess_results": hook,
        }
    )
    results_out, _, show_group_table, _ = _consolidate(group, _weighted_results())
    assert results_out["nohook"]["acc,none"] == 0.72
    assert results_out["nohook"]["marker,none"] == 5
    assert show_group_table is True


def test_hook_receives_samples_and_output_dir():
    received = {}

    def hook(group_name, subtasks, results, samples=None, output_dir=None, group_config=None, **kwargs):
        received["group_name"] = group_name
        received["subtasks"] = subtasks
        received["samples"] = samples
        received["output_dir"] = output_dir
        received["group_config"] = group_config
        return {"extra,none": 2}

    group = ConfigurableGroup(config={"group": "nohook", "task": ["t1", "t2"], "postprocess_results": hook, "metadata": {"version": "0.2"}})
    samples = {"t1": [{"doc_id": 0}], "t2": [{"doc_id": 1}]}
    results_out, versions_out, show_group_table, _ = _consolidate(group, _weighted_results(), samples=samples, output_dir="/tmp/out")
    assert received["group_name"] == "nohook"
    assert set(received["subtasks"]) == {"t1", "t2"}
    assert received["samples"] == samples
    assert received["output_dir"] == "/tmp/out"
    assert received["group_config"]["metadata"]["version"] == "0.2"
    assert results_out["nohook"]["extra,none"] == 2
    assert " " not in results_out["nohook"]
    assert show_group_table is True
    assert versions_out["nohook"] == "0.2"


def test_failing_hook_is_called_once_and_leaves_default_metrics():
    calls = {"n": 0}

    def bad_hook(group_name, subtasks, results, **kwargs):
        calls["n"] += 1
        raise TypeError("hook failed internally")

    group = ConfigurableGroup(
        config={
            "group": "nohook",
            "task": ["t1", "t2"],
            "aggregate_metric_list": AGGREGATE_METRIC_LIST,
            "postprocess_results": bad_hook,
        }
    )
    results_out, _, _, _ = _consolidate(group, _weighted_results())
    assert calls["n"] == 1
    assert results_out["nohook"]["acc,none"] == 0.72
    assert "marker,none" not in results_out["nohook"]


def test_signature_mismatch_does_not_call_hook():
    calls = {"n": 0}

    def bad_signature(only_arg, another):
        calls["n"] += 1
        return {"marker,none": 1}

    group = ConfigurableGroup(config={"group": "nohook", "task": ["t1", "t2"], "aggregate_metric_list": AGGREGATE_METRIC_LIST, "postprocess_results": bad_signature})
    results_out, _, _, _ = _consolidate(group, _weighted_results())
    assert calls["n"] == 0
    assert results_out["nohook"]["acc,none"] == 0.72


def test_hook_via_string_import_path():
    group = ConfigurableGroup(
        config={
            "group": "nohook",
            "task": ["t1", "t2"],
            "postprocess_results": "test.eval.test_group_postprocess.marker_hook",
        }
    )
    assert group.config["postprocess_results"] == "test.eval.test_group_postprocess.marker_hook"
    assert _resolve_group_postprocess_callable(group._config.postprocess_results) is marker_hook
    results_out, _, show_group_table, _ = _consolidate(group, _weighted_results())
    assert results_out["nohook"]["marker,none"] == 1
    assert show_group_table is True


def test_hook_via_yaml_function_reads_callable_from_config_object():
    yaml_content = textwrap.dedent(
        """
        group: nohook
        task:
          - t1
          - t2
        postprocess_results: !function test.eval.test_group_postprocess._yaml_hook
        """
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "group.yaml")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write(yaml_content)
        config = load_yaml_config(yaml_path=yaml_path, mode="full")
        group = ConfigurableGroup(config=config)
        assert callable(group._config.postprocess_results)
        assert isinstance(group.config["postprocess_results"], str)
        results_out, _, _, _ = _consolidate(group, _weighted_results())
        assert results_out["nohook"]["from_yaml,none"] == 3


def test_hook_none_and_empty_dict_do_not_crash():
    for hook in (lambda group_name, subtasks, results, **kwargs: None, lambda group_name, subtasks, results, **kwargs: {}):
        group = ConfigurableGroup(config={"group": "plain", "task": ["t1", "t2"], "postprocess_results": hook})
        results = {
            "t1": {"alias": "t1", "acc,none": 0.5, "samples": 1},
            "t2": {"alias": "t2", "acc,none": 0.5, "samples": 1},
            "plain": {"alias": "plain"},
        }
        results_out, _, _, _ = _consolidate(group, results)
        assert results_out["plain"][" "] == " "
        assert "marker,none" not in results_out["plain"]


def test_sitebench_merge_results_combines_subtask_samples():
    samples = {
        "site_bench_image": [
            {
                "accuracy": {"overall": 1, "total": 1},
                "chance_adjusted_acc": {"overall": 0.5, "total": 0.5},
                "doc": {"category": "counting & existence"},
            }
        ],
        "site_bench_video": [
            {
                "accuracy": {"overall": 0, "total": 1},
                "chance_adjusted_acc": {"overall": 0.0, "total": 0.5},
                "doc": {"category": "counting & existence"},
            }
        ],
    }
    merged = sitebench_merge_results(
        group_name="sitebench",
        subtask_names=["site_bench_image", "site_bench_video"],
        samples=samples,
    )
    assert merged["accuracy,none"] == 50.0
    assert merged["chance_adjusted_acc,none"] == 50.0
    assert merged["sitebench_accuracy,none"] == 50.0

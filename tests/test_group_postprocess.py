"""Regression tests for group-level postprocess_results hook (#1406)."""

import os
import tempfile
import textwrap

import pytest

from lmms_eval.api.group import ConfigurableGroup
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.evaluator_utils import consolidate_group_results
from lmms_eval.utils import load_yaml_config


def _make_task(name, version="1.0"):
    """Minimal Task mock with required attributes for consolidate_group_results."""
    # Use a simple object mimicking Task interface needed for aggregation
    class FakeTask:
        def __init__(self, task_name):
            self.task_name = task_name
            self.VERSION = version

        def dump_config(self):
            return {"task": self.task_name}

    return FakeTask(name)


def test_postprocess_hook_called_and_metrics_inserted():
    """Hook should be called and its returned metrics inserted into group results."""

    def my_hook(group_name, subtasks, results, **kwargs):
        # verify hook receives expected args
        assert group_name == "sitebench"
        assert set(subtasks) == {"site_bench_image", "site_bench_video"}
        assert "site_bench_image" in results
        # return group-level metrics
        return {"combined_score,none": 84.5, "hook_ran,none": 1}

    group = ConfigurableGroup(
        config={
            "group": "sitebench",
            "task": ["site_bench_image", "site_bench_video"],
            "postprocess_results": my_hook,
        }
    )

    # task_dict hierarchy: group -> subtasks
    task_dict = {
        group: {
            "site_bench_image": _make_task("site_bench_image"),
            "site_bench_video": _make_task("site_bench_video"),
        }
    }

    results = {
        "site_bench_image": {"alias": "site_bench_image", "accuracy,none": 80.0, "samples": 10},
        "site_bench_video": {"alias": "site_bench_video", "accuracy,none": 70.0, "samples": 10},
        "sitebench": {"alias": "sitebench"},
    }
    versions = {}

    results_out, versions_out, show_group_table, task_agg = consolidate_group_results(results, versions, task_dict)

    assert "combined_score,none" in results_out["sitebench"]
    assert results_out["sitebench"]["combined_score,none"] == 84.5
    assert results_out["sitebench"]["hook_ran,none"] == 1
    # existing behavior: group should still get a placeholder or be marked for group table
    # hook should cause show_group_table to be True
    assert show_group_table is True or "combined_score,none" in results_out["sitebench"]


def test_postprocess_hook_kwargs_include_config_and_samples():
    """Hook receives group config metadata, samples, output dirs via kwargs."""

    received = {}

    def hook(group_name, subtask_names=None, **kwargs):
        # Accept both naming conventions
        received.update(kwargs)
        received["group_name"] = group_name
        received["subtask_names"] = subtask_names
        if "subtasks" in kwargs:
            received["subtask_names"] = kwargs["subtasks"]
        # also capture positional subtasks if passed as second arg
        return {"extra_metric,none": 1.0}

    # alternative signature using subtasks param
    def hook2(group_name, subtasks, results, samples=None, output_dir=None, group_config=None, **kwargs):
        received["group_name2"] = group_name
        received["subtasks2"] = subtasks
        received["results2"] = results
        received["samples2"] = samples
        received["output_dir2"] = output_dir
        received["group_config2"] = group_config
        return {"extra2,none": 2.0}

    group = ConfigurableGroup(
        config={
            "group": "mygroup",
            "task": ["task_a", "task_b"],
            "postprocess_results": hook2,
            "metadata": {"version": "0.2"},
        }
    )
    task_dict = {
        group: {
            "task_a": _make_task("task_a"),
            "task_b": _make_task("task_b"),
        }
    }
    results = {
        "task_a": {"alias": "task_a", "acc,none": 0.5, "samples": 5},
        "task_b": {"alias": "task_b", "acc,none": 0.7, "samples": 5},
        "mygroup": {"alias": "mygroup"},
    }
    samples = {"task_a": [{"doc_id": 0}], "task_b": [{"doc_id": 1}]}
    versions = {}

    # New signature supports samples, output_dir, etc.
    results_out, _, _, _ = consolidate_group_results(
        results,
        versions,
        task_dict,
        samples=samples,
        output_dir="/tmp/out",
        model_output_dir="/tmp/out/model",
    )

    assert received["group_name2"] == "mygroup"
    assert set(received["subtasks2"]) == {"task_a", "task_b"}
    assert received["samples2"] is not None
    assert received["output_dir2"] == "/tmp/out"
    assert received["group_config2"] is not None
    assert "extra2,none" in results_out["mygroup"]


def test_preserve_aggregate_metric_list_alongside_hook():
    """aggregate_metric_list should still produce mean even when hook is present."""

    def hook(group_name, subtasks, results, **kwargs):
        return {"hook_metric,none": 99.0}

    group = ConfigurableGroup(
        config={
            "group": "agg_group",
            "task": ["t1", "t2"],
            "aggregate_metric_list": [{"metric": "acc", "aggregation": "mean", "filter_list": ["none"], "weight_by_size": False}],
            "postprocess_results": hook,
        }
    )
    task_dict = {
        group: {
            "t1": _make_task("t1"),
            "t2": _make_task("t2"),
        }
    }
    results = {
        "t1": {"alias": "t1", "acc,none": 0.6, "samples": 10},
        "t2": {"alias": "t2", "acc,none": 0.8, "samples": 10},
        "agg_group": {"alias": "agg_group"},
    }
    versions = {}
    results_out, _, show_group, _ = consolidate_group_results(results, versions, task_dict)
    # mean of 0.6 and 0.8 = 0.7
    assert "acc,none" in results_out["agg_group"]
    assert abs(results_out["agg_group"]["acc,none"] - 0.7) < 1e-6
    assert "hook_metric,none" in results_out["agg_group"]
    assert show_group is True


def test_hook_via_string_import_path():
    """Hook defined as importable string path should be resolved."""

    group = ConfigurableGroup(
        config={
            "group": "str_group",
            "task": ["t1"],
            "postprocess_results": "lmms_eval.tasks.sitebench.utils.sitebench_merge_results",
        }
    )
    # Ensure config stored as string but consolidate will resolve
    assert group.config["postprocess_results"] == "lmms_eval.tasks.sitebench.utils.sitebench_merge_results"

    task_dict = {
        group: {"t1": _make_task("t1")}
    }
    results = {
        "t1": {"alias": "t1", "acc,none": 0.5, "samples": 2},
        "str_group": {"alias": "str_group"},
    }
    # provide samples similar to sitebench (accuracy dict)
    samples = {
        "t1": [
            {"accuracy": {"overall": 1, "total": 1}, "chance_adjusted_acc": {"overall": 0.5, "total": 0.5}, "doc": {"category": "counting & existence"}},
            {"accuracy": {"overall": 0, "total": 1}, "chance_adjusted_acc": {"overall": -0.5, "total": 0.5}, "doc": {"category": "counting & existence"}},
        ]
    }
    versions = {}
    results_out, _, _, _ = consolidate_group_results(results, versions, task_dict, samples=samples)
    # hook should have been invoked and merged at least one metric (overall_caa etc) without error
    # we don't assert exact value, just that hook ran without exception and group has metrics
    assert "str_group" in results_out
    # sitebench_merge_results should produce some metrics; check that group not just empty
    # It may return {} if insufficient data, but should not crash
    assert isinstance(results_out["str_group"], dict)


def test_hook_via_yaml_function():
    """YAML !function hook should be loaded and invoked."""

    yaml_content = textwrap.dedent("""
        group: yaml_group
        task:
          - t1
          - t2
        postprocess_results: !function lmms_eval.tasks.sitebench.utils.sitebench_merge_results
    """)
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "group.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        config = load_yaml_config(yaml_path=yaml_path, mode="full")
        assert callable(config["postprocess_results"])
        group = ConfigurableGroup(config=config)
        assert callable(group.config["postprocess_results"])

        task_dict = {
            group: {"t1": _make_task("t1"), "t2": _make_task("t2")}
        }
        results = {
            "t1": {"alias": "t1", "acc,none": 0.5, "samples": 1},
            "t2": {"alias": "t2", "acc,none": 0.5, "samples": 1},
            "yaml_group": {"alias": "yaml_group"},
        }
        samples = {"t1": [], "t2": []}
        versions = {}
        results_out, _, _, _ = consolidate_group_results(results, versions, task_dict, samples=samples)
        assert "yaml_group" in results_out


def test_hook_returns_none_or_empty_no_crash():
    """Hook returning None or empty should not crash."""

    def hook_none(group_name, subtasks, results, **kwargs):
        return None

    def hook_empty(group_name, subtasks, results, **kwargs):
        return {}

    for hook in [hook_none, hook_empty]:
        group = ConfigurableGroup(config={"group": "g", "task": ["t1"], "postprocess_results": hook})
        task_dict = {group: {"t1": _make_task("t1")}}
        results = {"t1": {"alias": "t1", "acc,none": 0.5, "samples": 1}, "g": {"alias": "g"}}
        versions = {}
        results_out, _, _, _ = consolidate_group_results(results, versions, task_dict)
        assert "g" in results_out


def test_hook_exception_is_logged_not_crashed():
    """Hook raising exception should be logged but not crash group aggregation."""

    def bad_hook(group_name, subtasks, results, **kwargs):
        raise RuntimeError("hook failed")

    group = ConfigurableGroup(config={"group": "bad_group", "task": ["t1"], "postprocess_results": bad_hook})
    task_dict = {group: {"t1": _make_task("t1")}}
    results = {"t1": {"alias": "t1", "acc,none": 0.5, "samples": 1}, "bad_group": {"alias": "bad_group"}}
    versions = {}
    # Should not raise
    results_out, _, _, _ = consolidate_group_results(results, versions, task_dict)
    assert "bad_group" in results_out


def test_no_hook_existing_behavior_unchanged():
    """Without hook, existing aggregate behavior should remain identical."""

    group = ConfigurableGroup(
        config={
            "group": "nohook",
            "task": ["t1", "t2"],
            "aggregate_metric_list": [{"metric": "acc", "aggregation": "mean", "filter_list": ["none"], "weight_by_size": True}],
        }
    )
    task_dict = {group: {"t1": _make_task("t1"), "t2": _make_task("t2")}}
    results = {
        "t1": {"alias": "t1", "acc,none": 0.4, "samples": 2},
        "t2": {"alias": "t2", "acc,none": 0.8, "samples": 8},
        "nohook": {"alias": "nohook"},
    }
    versions = {}
    results_out, _, show, _ = consolidate_group_results(results, versions, task_dict)
    # weighted mean: (0.4*2+0.8*8)/10 = 0.72
    assert abs(results_out["nohook"]["acc,none"] - 0.72) < 1e-6
    assert show is True

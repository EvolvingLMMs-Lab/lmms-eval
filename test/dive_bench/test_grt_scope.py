"""Whole-batch scope checks on actual registered wrappers, with CPU-only parents."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from lmms_eval.api import registry
from lmms_eval.api.instance import Instance
from lmms_eval.models.model_utils.grt import runtime
from lmms_eval.models.model_utils.grt.profile import command, load_profiles
from lmms_eval.models.model_utils.grt.scope import SUPPORTED_GRT_TASKS, EducationalGRTScope, preflight_grt_requests, require_grt_worker_tasks

WRAPPERS = [
    ("grt_llava_hf", "llava_hf", "LlavaHf", "GRTLlavaHf"),
    ("grt_qwen2_5_vl", "qwen2_5_vl", "Qwen2_5_VL", "GRTQwen2_5VL"),
    ("grt_qwen2_5_vl_floor", "qwen_dual_route", "Qwen2_5_VL_DualRouteFloor", "GRTQwen2_5VLFloor"),
]
ENTRY_POINTS = ["generate_until", "loglikelihood", "generate_until_multi_round"]
HELD_TASKS = ["dive_bench_high_motion_high_fps", "dive_bench_high_motion_high_fps_preview1000", "densevideo_highmotion"]
ROOT = Path(__file__).resolve().parents[2]


def request(task="densevideo", entry="generate_until"):
    return Instance(request_type=entry, arguments=("synthetic prompt", {}, mock.Mock(), 0, task, "test"), idx=0, metadata={"task": task, "doc_id": 0, "repeats": 1})


@pytest.fixture(params=WRAPPERS, ids=lambda entry: entry[0])
def adapter(request, monkeypatch):
    """Execute each real adapter module while replacing only its frozen parent."""
    filename, parent_module, parent_name, class_name = request.param
    calls = mock.Mock()

    class Parent:
        def __init__(self, **kwargs):
            calls.initialize(kwargs)

        def generate_until(self, requests):
            calls.backend("generate_until", requests)
            return requests

        def loglikelihood(self, requests):
            calls.backend("loglikelihood", requests)
            return requests

        def generate_until_multi_round(self, requests):
            calls.backend("generate_until_multi_round", requests)
            return requests

    module_name = "lmms_eval.models.model_utils.grt." + parent_module
    parent = ModuleType(module_name)
    setattr(parent, parent_name, Parent)
    monkeypatch.setitem(sys.modules, module_name, parent)
    monkeypatch.setattr(registry, "register_model", lambda _: lambda cls: cls)
    monkeypatch.setattr(runtime, "require_grt_runtime", calls.runtime)
    path = ROOT / "lmms_eval/models/simple" / (filename + ".py")
    spec = importlib.util.spec_from_file_location("scope_test_" + filename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, class_name)(marker="unchanged")
    calls.runtime.assert_called_once_with()
    calls.initialize.assert_called_once_with({"marker": "unchanged"})
    return SimpleNamespace(model=model, calls=calls)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("task", HELD_TASKS + ["other_task", "densevideo*", None, 42, [], ""])
def test_registered_adapters_reject_entire_mixed_batch_before_parent(adapter, entry, task):
    good = request(entry=entry)
    bad = request(task=task, entry=entry)
    method = getattr(adapter.model, entry)
    assert method.__func__ is getattr(EducationalGRTScope, entry)
    with pytest.raises(ValueError, match="Educational"):
        method(iter([good, bad]))
    adapter.calls.backend.assert_not_called()
    good.args[2].assert_not_called()
    bad.args[2].assert_not_called()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_educational_aliases_preserve_order_identity_and_single_pass_iterables(adapter, entry):
    expected = [request(task, entry) for task in ("dive_bench_educational_high_fps", "densevideo")]

    class Once:
        traversals = 0

        def __iter__(self):
            self.traversals += 1
            assert self.traversals == 1
            yield from expected

    source = Once()
    actual = getattr(adapter.model, entry)(source)
    assert source.traversals == 1
    assert len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected))
    adapter.calls.backend.assert_called_once_with(entry, actual)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("bad", [None, object(), SimpleNamespace(args=()), SimpleNamespace(args=(1, 2, 3, 4, "densevideo")), SimpleNamespace(args="densevideo")])
def test_malformed_later_request_rejects_whole_batch(adapter, entry, bad):
    with pytest.raises(ValueError, match="six-field"):
        getattr(adapter.model, entry)([request(entry=entry), bad])
    adapter.calls.backend.assert_not_called()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("field", ["metadata", "task_name"])
def test_conflicting_request_identity_is_rejected(adapter, entry, field):
    bad = request(entry=entry)
    setattr(bad, field, {"task": HELD_TASKS[0]} if field == "metadata" else HELD_TASKS[0])
    with pytest.raises(ValueError, match="identities disagree"):
        getattr(adapter.model, entry)([request(entry=entry), bad])
    adapter.calls.backend.assert_not_called()


def test_noniterable_batch_is_rejected():
    with pytest.raises(ValueError, match="iterable"):
        preflight_grt_requests(None)


def test_generator_failure_cannot_send_partial_batch_to_backend(adapter):
    def broken():
        yield request()
        raise RuntimeError("synthetic iterator failure")

    with pytest.raises(RuntimeError, match="iterator failure"):
        adapter.model.generate_until(broken())
    adapter.calls.backend.assert_not_called()


@pytest.mark.parametrize("profile", ["route31", "qwen3", "qwen7"])
@pytest.mark.parametrize("role", ["base", "all", "candidate"])
def test_all_frozen_profile_commands_remain_accepted(profile, role, tmp_path):
    args = command(profile, role, tmp_path / "unused")
    require_grt_worker_tasks(args[3:])
    assert load_profiles()["profiles"][profile]["task"] == "dive_bench_educational_high_fps"
    assert SUPPORTED_GRT_TASKS == {"dive_bench_educational_high_fps", "densevideo"}
    assert not (tmp_path / "unused").exists()

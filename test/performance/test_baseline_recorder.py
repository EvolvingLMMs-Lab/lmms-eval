import json
import re
from types import MappingProxyType, SimpleNamespace

import pytest

import lmms_eval._performance.recorder as recorder_module
import lmms_eval.performance as performance
from lmms_eval._performance.provenance import BaselineProvenance
from lmms_eval.performance import BaselinePerformanceRecorder

TOP_LEVEL = {"schema_version", "record_kind", "source_commit", "source_tree_digest", "legacy_invocation", "environment_lock_digest", "hardware", "cache_state", "repetition", "phases", "counters", "resources"}
FORBIDDEN_KEYS = {"eval_spec", "identity", "intent_id", "resolved_eval_id", "spec_digest", "runtime_id", "run_id", "attempt_id"}


def _all_keys(value):
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return (set(value) if isinstance(value, dict) else set()).union(*map(_all_keys, items))


@pytest.fixture
def recorder_factory(monkeypatch, tmp_path):
    fixed = BaselineProvenance("a" * 40, "sha256:" + "b" * 64, "sha256:" + "c" * 64, "fixture-hardware")
    monkeypatch.setattr(recorder_module, "capture_baseline_provenance", lambda root: fixed)

    def make(*, started=False, finished=False, **overrides):
        value = BaselinePerformanceRecorder.capture(
            repo_root=tmp_path,
            legacy_arguments=overrides.pop("legacy_arguments", {"model": "dummy"}),
            cache_state=overrides.pop("cache_state", "disabled"),
            repetition=overrides.pop("repetition", {"suite_id": "suite", "case_id": "case", "repetition_index": 0, "warmup": False}),
            digest_legacy_arguments=overrides.pop("digest_legacy_arguments", False),
        )
        assert not overrides
        if started or finished:
            value.start()
        if finished:
            value.finish()
        return value

    return make


def test_baseline_record_has_exact_v1_shape(monkeypatch, tmp_path, recorder_factory):
    ticks = iter((100, 110, 120, 130, 150, 200))
    monkeypatch.setattr(recorder_module.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(recorder_module, "peak_host_rss_bytes", lambda: 4096)
    recorder = recorder_factory(legacy_arguments={"model": "dummy", "limit": "0.5"}, repetition={"suite_id": "hermetic-cpu-dummy-v1", "case_id": "cache-disabled", "repetition_index": 0, "warmup": False})
    recorder.start()
    with recorder.phase("model_load"):
        recorder.increment("responses", 2)
    with recorder.phase("model_load"):
        recorder.increment("responses")
    recorder.set_resource("model_load_reused", False)
    recorder.finish()
    record = recorder.to_record()
    assert set(record) == TOP_LEVEL
    assert (record["schema_version"], record["record_kind"]) == (1, "baseline")
    assert record["legacy_invocation"] == {"kind": "normalized", "arguments": {"model": "dummy", "limit": "0.5"}}
    assert record["phases"] == [{"name": "model_load", "owner": "worker", "duration_ns": 30, "overlapped": False}]
    assert record["counters"] == {"failures": 0, "responses": 3}
    assert record["resources"] == {"model_load_reused": False, "end_to_end_duration_ns": 100, "peak_host_rss_bytes": 4096}
    assert FORBIDDEN_KEYS.isdisjoint(_all_keys(record))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", record["source_tree_digest"])
    output = recorder.write_json(tmp_path / "baseline.json")
    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert not output.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "repetition",
    [
        {"suite_id": "suite", "case_id": "case", "repetition_index": 0},
        {"suite_id": "suite", "case_id": "case", "repetition_index": 0, "warmup": False, "run": 1},
        {"suite_id": "", "case_id": "case", "repetition_index": 0, "warmup": False},
        {"suite_id": "suite", "case_id": "", "repetition_index": 0, "warmup": False},
        {"suite_id": "suite", "case_id": "case", "repetition_index": True, "warmup": False},
        {"suite_id": "suite", "case_id": "case", "repetition_index": -1, "warmup": False},
        {"suite_id": "suite", "case_id": "case", "repetition_index": 2**53, "warmup": False},
        {"suite_id": "suite", "case_id": "case", "repetition_index": 2**63, "warmup": False},
        {"suite_id": "suite", "case_id": "case", "repetition_index": 0, "warmup": 0},
        {"suite_id": "suite", "case_id": "case", "repetition_index": 0, "warmup": None},
    ],
)
def test_capture_rejects_invalid_repetition(recorder_factory, repetition):
    with pytest.raises((TypeError, ValueError), match="repetition"):
        recorder_factory(repetition=repetition)


@pytest.mark.parametrize(("key", "value"), [(1, "non-string-key"), ("bad", "\ud800"), ("bad", 2**53), ("bad", -(2**53)), ("bad", 2**63), ("bad", -(2**63) - 1), ("bad", float("nan")), ("bad", float("inf")), ("bad", -0.0), ("bad", object())])
def test_to_record_rejects_invalid_v1_json_values(recorder_factory, key, value):
    recorder = recorder_factory(finished=True)
    recorder.resources[key] = value
    with pytest.raises((TypeError, ValueError)):
        recorder.to_record()


def test_to_record_allows_json_null_inside_open_resource_map(recorder_factory):
    recorder = recorder_factory(finished=True)
    recorder.set_resource("optional_measurement", None)
    assert recorder.to_record()["resources"]["optional_measurement"] is None


def test_to_record_rejects_non_concrete_json_mapping(recorder_factory):
    recorder = recorder_factory(finished=True)
    recorder.set_resource("nested", MappingProxyType({"value": 1}))
    with pytest.raises(TypeError, match="concrete dict"):
        recorder.to_record()


def test_to_record_returns_deeply_detached_snapshot(recorder_factory, tmp_path):
    recorder = recorder_factory(finished=True, legacy_arguments={"nested": {"value": "stable"}})
    recorder.counters["nested"] = {"value": "stable"}
    recorder.set_resource("nested", {"values": [1]})
    first = recorder.to_record()
    first["legacy_invocation"]["arguments"]["nested"]["value"] = first["repetition"]["suite_id"] = first["counters"]["nested"]["value"] = "mutated"
    first["resources"]["nested"]["values"].append(2)
    second = recorder.to_record()
    assert (second["legacy_invocation"]["arguments"]["nested"], second["repetition"]["suite_id"]) == ({"value": "stable"}, "suite")
    assert (second["counters"]["nested"], second["resources"]["nested"]) == ({"value": "stable"}, {"values": [1]})
    assert json.loads(recorder.write_json(tmp_path / "snapshot.json").read_text()) == second


@pytest.mark.parametrize("phase", [("future", "worker", 1, False), ("model_load", "invalid", 1, False), *(("model_load", "worker", value, False) for value in (True, -1, 2**53)), ("model_load", "worker", 1, 0)])
def test_to_record_rejects_invalid_phase_shape(recorder_factory, phase):
    name, owner, duration_ns, overlapped = phase
    recorder = recorder_factory(finished=True)
    recorder._phases[name] = SimpleNamespace(owner=owner, duration_ns=duration_ns, overlapped=overlapped)
    with pytest.raises((TypeError, ValueError), match="phase"):
        recorder.to_record()


def test_phase_rejects_inconsistent_metadata_before_body_side_effects(recorder_factory):
    recorder = recorder_factory(started=True)
    with recorder.phase("score", owner="worker"):
        pass
    entered = False
    with pytest.raises(ValueError, match="inconsistent phase metadata"):
        with recorder.phase("score", owner="control"):
            entered = True
    assert entered is False


@pytest.mark.parametrize("overlapped", [0, None])
def test_phase_rejects_non_boolean_overlap_before_body_and_state(recorder_factory, overlapped):
    recorder = recorder_factory(started=True)
    entered = False
    with pytest.raises(TypeError, match="overlapped"):
        with recorder.phase("score", overlapped=overlapped):
            entered = True
    assert (entered, recorder._phases) == (False, {})


def test_phase_preserves_body_exception_and_counts_failure(recorder_factory):
    recorder = recorder_factory(started=True)
    with pytest.raises(LookupError, match="body failure"):
        with recorder.phase("score"):
            raise LookupError("body failure")
    assert recorder.counters["failures"] == 1
    recorder.finish()
    assert recorder.to_record()["counters"]["failures"] == 1


def test_active_phase_blocks_finish_and_record_emission(recorder_factory, tmp_path):
    recorder = recorder_factory(started=True)
    with recorder.phase("inference"):
        for action in (recorder.finish, recorder.to_record, lambda: recorder.write_json(tmp_path / "invalid.json")):
            with pytest.raises(RuntimeError, match="active phase"):
                action()
        assert "end_to_end_duration_ns" not in recorder.resources
    recorder.finish()
    assert recorder.to_record()["resources"]["end_to_end_duration_ns"] >= 0


def test_recorder_rejects_invalid_start_finish_state(recorder_factory):
    recorder = recorder_factory()
    for action, match in ((recorder.finish, "started"), (recorder.to_record, "finished")):
        with pytest.raises(RuntimeError, match=match):
            action()
    with pytest.raises(RuntimeError, match="started"):
        with recorder.phase("score"):
            pass
    recorder.start()
    with pytest.raises(RuntimeError, match="already started"):
        recorder.start()
    recorder.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        recorder.finish()
    with pytest.raises(RuntimeError, match="finished"):
        with recorder.phase("score"):
            pass


def test_baseline_record_rejects_structural_future_identity_but_allows_runtime_observation(recorder_factory):
    recorder = recorder_factory(finished=True)
    recorder.set_resource("runtime", {"python": "3.10"})
    assert recorder.to_record()["resources"]["runtime"] == {"python": "3.10"}
    recorder.set_resource("identity", {"spec_digest": "sha256:" + "d" * 64})
    with pytest.raises(ValueError, match="future identity"):
        recorder.to_record()


def test_public_performance_facade_is_narrow():
    assert performance.__all__ == ["BaselinePerformanceRecorder", "redact_secrets"]
    assert performance.BaselinePerformanceRecorder is BaselinePerformanceRecorder
    assert callable(performance.redact_secrets)

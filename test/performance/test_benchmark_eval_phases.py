import json
import socket

import pytest

import tools.benchmark_eval_phases as benchmark
from tools.benchmark_eval_phases import (
    _build_parser,
    _sha256,
    _summarize,
    _versions,
    run_suite,
)

_FUTURE_IDENTITIES = {"eval_spec", "identity", "intent_id", "resolved_eval_id", "spec_digest", "runtime_id", "run_id", "attempt_id"}


def _all_keys(value):
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return (set(value) if isinstance(value, dict) else set()).union(*map(_all_keys, items))


def _records(paths):
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_hermetic_suite_emits_three_cache_cases_with_equal_results(monkeypatch, tmp_path):
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network access")))

    records = _records(run_suite(tmp_path, warmups=1, measured_repetitions=1, digest_legacy_arguments=False))

    assert len(records) == 6
    assert {record["repetition"]["case_id"] for record in records} == {"cache-disabled", "cache-cold", "cache-warm"}
    assert {record["cache_state"] for record in records} == {"disabled", "cold", "warm"}
    assert len({record["resources"]["correctness_digest"] for record in records}) == 1
    for record in records:
        assert {record["counters"][name] for name in ("selected_documents", "built_instances", "responses", "normalized_outputs", "scored_documents")} == {4}
        assert record["counters"]["artifact_files"] == 3
        assert record["counters"]["batches"] == record["counters"]["inference_dispatches"]
        assert {"request_cache_hits", "request_cache_misses"}.isdisjoint(record["counters"])
        assert record["counters"]["end_to_end_scored_documents_per_second"] > 0
        assert "artifact_stage" in {phase["name"] for phase in record["phases"]}
        assert record["resources"]["revisions"]["model"]["id"] == "dummy"
        assert record["resources"]["outcome"] == "completed"
        assert set(record["resources"]["backend_versions"]) == {"python", "lmms_eval", "datasets", "torch"}
        assert set(record["resources"]["unmeasured_phases"]) == {"queue_wait", "preprocess"}
        assert _FUTURE_IDENTITIES.isdisjoint(_all_keys(record))


def test_existing_record_is_rejected_before_evaluation(monkeypatch, tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "cache-disabled-000-warmup.json").write_text("occupied")
    monkeypatch.setattr("tools.benchmark_eval_phases.simple_evaluate", lambda **kwargs: pytest.fail("evaluation started"))

    with pytest.raises(FileExistsError, match="cache-disabled-000-warmup"):
        run_suite(tmp_path, warmups=1, measured_repetitions=1)


def test_artifact_failure_record_exposes_only_stable_stage_and_type(monkeypatch, tmp_path):
    secret = "credential-value-must-not-escape"

    def fail(*args, **kwargs):
        raise LookupError(secret)

    monkeypatch.setattr("tools.benchmark_eval_phases.EvaluationTracker.save_results_samples", fail)
    with pytest.raises(LookupError, match=secret):
        run_suite(tmp_path, warmups=0, measured_repetitions=1)

    failed = json.loads(next((tmp_path / "records").glob("*-failed.json")).read_text(encoding="utf-8"))
    assert failed["resources"]["failure"] == {"stage": "artifact_stage", "type": "LookupError"}
    assert failed["resources"]["outcome"] == "failed"
    assert failed["resources"]["partial_artifacts"]["policy"] == "preserve"
    assert failed["resources"]["partial_artifacts"]["files"] >= 1
    assert failed["counters"]["failures"] == 1
    assert secret not in json.dumps(failed)
    monkeypatch.setattr("tools.benchmark_eval_phases.BaselinePerformanceRecorder.write_json", lambda *args: (_ for _ in ()).throw(OSError("reporting failed")))
    with pytest.raises(LookupError, match=secret):
        run_suite(tmp_path / "writer-failure", warmups=0, measured_repetitions=1)


def test_warm_case_rejects_a_broken_request_cache_load(monkeypatch, tmp_path):
    monkeypatch.setattr("lmms_eval.api.task.load_from_cache", lambda file_name: None)
    with pytest.raises(RuntimeError, match="cache lifecycle"):
        run_suite(tmp_path, warmups=0, measured_repetitions=1)
    failed = json.loads(next((tmp_path / "records").glob("*-failed.json")).read_text(encoding="utf-8"))
    assert (failed["resources"]["failure"]["stage"], failed["resources"]["outcome"]) == ("cache_validation", "failed")


def test_repetition_atomically_claims_its_artifact_root(monkeypatch, tmp_path):
    (tmp_path / "artifacts" / "cache-disabled-000-measured").mkdir(parents=True)
    monkeypatch.setattr(benchmark, "simple_evaluate", lambda **kwargs: pytest.fail("evaluation started"))
    with pytest.raises(FileExistsError):
        benchmark._run_repetition(tmp_path, "cache-disabled", 0, False, False)


def test_cache_validation_runs_outside_the_timed_window(monkeypatch, tmp_path):
    events = []
    original_snapshot = benchmark._cache_snapshot
    original_start, original_finish = benchmark.BaselinePerformanceRecorder.start, benchmark.BaselinePerformanceRecorder.finish
    monkeypatch.setattr(benchmark.request_cache, "PATH", str(tmp_path / "cache"))
    monkeypatch.setattr(benchmark, "_cache_snapshot", lambda: events.append("snapshot") or original_snapshot())
    monkeypatch.setattr(benchmark.BaselinePerformanceRecorder, "start", lambda self: events.append("start") or original_start(self))
    monkeypatch.setattr(benchmark.BaselinePerformanceRecorder, "finish", lambda self: events.append("finish") or original_finish(self))

    benchmark._run_repetition(tmp_path, "cache-disabled", 0, False, False)

    assert events == ["snapshot", "start", "finish", "snapshot"]


def test_unreadable_pretiming_cache_emits_failed_record(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / f"broken{benchmark.request_cache.FILE_SUFFIX}").write_bytes(b"not a pickle")
    monkeypatch.setattr(benchmark.request_cache, "PATH", str(cache_root))
    with pytest.raises(RuntimeError, match="unreadable"):
        benchmark._run_repetition(tmp_path, "cache-warm", 0, False, False)
    failed = json.loads(next((tmp_path / "records").glob("*-failed.json")).read_text())
    assert failed["resources"]["failure"] == {"stage": "cache_validation", "type": "RuntimeError"}


def test_summary_excludes_warmups_and_uses_nearest_rank_p95():
    def record(value, warmup=False):
        return {
            "repetition": {"case_id": "case", "warmup": warmup},
            "phases": [{"name": "inference", "duration_ns": value}],
            "resources": {"end_to_end_duration_ns": value, "peak_host_rss_bytes": value},
            "counters": {"end_to_end_scored_documents_per_second": value},
        }

    summary = _summarize([record(999, True), *map(record, [1, 2, 3, 4, 100])])

    assert summary["case"]["end_to_end_duration_ns"] == {"median": 3, "p95": 100, "pvariance": 1522}


def test_correctness_mismatch_emits_no_summary(monkeypatch, tmp_path, capsys):
    digests = iter(("sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64))

    def write_record(output_dir, case_id, index, warmup, digest_legacy_arguments):
        path = output_dir / "records" / f"{case_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"resources": {"correctness_digest": next(digests)}}), encoding="utf-8")
        return path

    monkeypatch.setattr(benchmark, "_evaluate", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, "_run_repetition", write_record)
    with pytest.raises(RuntimeError, match="correctness parity"):
        run_suite(tmp_path, warmups=0, measured_repetitions=1)
    assert capsys.readouterr().out == ""


def test_digest_uses_v1_domain_and_version_does_not_require_package_metadata(monkeypatch):
    with pytest.raises(ValueError, match="negative zero"):
        _sha256({"value": -0.0})
    monkeypatch.setattr(benchmark, "get_lmms_eval_cache_version", lambda: "source-revision")
    assert _versions()["lmms_eval"] == "source-revision"


def test_cli_requires_output_dir_and_has_stable_repetition_defaults():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--output-dir", "results"])
    assert (args.output_dir.name, args.warmups, args.measured_repetitions, args.digest_legacy_arguments) == ("results", 1, 5, False)


@pytest.mark.parametrize(("warmups", "measured"), [(-1, 1), (0, 0)])
def test_suite_rejects_invalid_repetition_counts(tmp_path, warmups, measured):
    with pytest.raises(ValueError):
        run_suite(tmp_path, warmups=warmups, measured_repetitions=measured)

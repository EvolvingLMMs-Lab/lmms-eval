#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import tempfile
from pathlib import Path

import datasets
import torch
from datasets import Dataset, DatasetDict

import lmms_eval.caching.cache as request_cache
from lmms_eval._performance.json_v1 import canonical_json_bytes
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.performance import BaselinePerformanceRecorder
from lmms_eval.utils import get_lmms_eval_cache_version

_SUITE_ID = "hermetic-cpu-dummy-v1"
_CASES = ("cache-disabled", "cache-cold", "cache-warm")
_TASK_SIZES = {"phase_baseline_small": 4, "phase_baseline_large": 10}


class _SyntheticTask(ConfigurableTask):
    def __init__(self, task_name: str, size: int) -> None:
        self._benchmark_documents = Dataset.from_list(_documents(task_name, size))
        super().__init__(
            config={
                "task": task_name,
                "dataset_path": None,
                "test_split": "test",
                "output_type": "generate_until",
                "doc_to_text": "question",
                "doc_to_target": "answer",
                "doc_to_visual": lambda doc: [],
                "generation_kwargs": {"max_new_tokens": 1},
                "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
            }
        )

    def download(self, dataset_kwargs=None) -> None:
        self.dataset = DatasetDict({"test": self._benchmark_documents})
        self.dataset_no_image = self.dataset


def _documents(task_name: str, size: int) -> list[dict[str, str]]:
    return [{"question": f"{task_name} question {index}", "answer": "A"} for index in range(size)]


def _make_tasks() -> list[ConfigurableTask]:
    return [_SyntheticTask(name, size) for name, size in _TASK_SIZES.items()]


def _legacy_arguments(case_id: str) -> dict[str, object]:
    return {
        "model": "dummy",
        "model_args": "response=A",
        "tasks": list(_TASK_SIZES),
        "limit": 2,
        "bootstrap_iters": 0,
        "log_samples": True,
        "cache_requests": case_id != "cache-disabled",
        "device": "cpu",
    }


def _evaluate(tasks, *, recorder=None, tracker=None, cache_requests_enabled: bool):
    return simple_evaluate(
        model="dummy",
        model_args="response=A",
        tasks=tasks,
        device="cpu",
        limit=2,
        bootstrap_iters=0,
        log_samples=True,
        cache_requests=cache_requests_enabled,
        evaluation_tracker=tracker,
        performance_recorder=recorder,
    )


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _correctness_projection(results: dict, samples: dict) -> dict:
    sample_projection = []
    for task_name in sorted(samples):
        for sample in sorted(samples[task_name], key=lambda item: item["doc_id"]):
            sample_projection.append(
                {
                    "task": task_name,
                    "doc_id": sample["doc_id"],
                    "raw_outputs": sample["resps"],
                    "normalized_outputs": sample["filtered_resps"],
                    "scores": {"exact_match": float(sample["exact_match"])},
                }
            )
    aggregate_projection = {task: float(metrics["exact_match,none"]) for task, metrics in sorted(results["results"].items())}
    return {"samples": sample_projection, "aggregate_metrics": aggregate_projection}


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "lmms_eval": get_lmms_eval_cache_version(),
        "datasets": datasets.__version__,
        "torch": str(torch.__version__),
    }


def _cache_snapshot() -> dict:
    root = Path(request_cache.PATH)
    paths = list(root.glob(f"*{request_cache.FILE_SUFFIX}")) if root.exists() else []
    if any(request_cache.load_from_cache(path.name.removesuffix(request_cache.FILE_SUFFIX)) is None for path in paths):
        raise RuntimeError("request cache lifecycle produced an unreadable file")
    return {path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).digest()) for path in paths}


def _resources(recorder: BaselinePerformanceRecorder) -> dict[str, object]:
    return {
        "model_load_reused": False,
        "batch_scope": "dummy model-method submission",
        "host_rss_scope": "process lifetime high-water",
        "peak_gpu_allocation_unavailable": "CPU-only workload",
        "reset_scope": "legacy model cleanup; no reuse health check",
        "phase_sum_may_exceed_end_to_end": False,
        "unavailable_counters": ["input_tokens", "output_tokens", "inference_tokens_per_second"],
        "unmeasured_phases": {
            "queue_wait": "legacy in-process benchmark has no EvaluationControl admission",
            "preprocess": "dummy performs no media/message conversion",
        },
        "revisions": {
            "model": {"id": "dummy", "revision": recorder.source_tree_digest},
            "datasets": {name: _sha256(_documents(name, size)) for name, size in _TASK_SIZES.items()},
        },
        "backend_versions": _versions(),
    }


def _run_repetition(output_dir: Path, case_id: str, index: int, warmup: bool, digest_legacy_arguments: bool) -> Path:
    label = "warmup" if warmup else "measured"
    stem = f"{case_id}-{index:03d}-{label}"
    record_path = output_dir / "records" / f"{stem}.json"
    failed_path = output_dir / "records" / f"{stem}-failed.json"
    artifact_root = output_dir / "artifacts" / stem
    artifact_root.mkdir(parents=True, exist_ok=False)
    recorder = BaselinePerformanceRecorder.capture(
        repo_root=Path(__file__).resolve().parents[1],
        legacy_arguments=_legacy_arguments(case_id),
        cache_state=case_id.removeprefix("cache-"),
        repetition={"suite_id": _SUITE_ID, "case_id": case_id, "repetition_index": index, "warmup": warmup},
        digest_legacy_arguments=digest_legacy_arguments,
    )
    for name, value in _resources(recorder).items():
        recorder.set_resource(name, value)
    stage = "cache_validation"
    started = False
    finished = False
    try:
        cache_before = _cache_snapshot()
        recorder.start()
        started = True
        stage = "task_resolution"
        with recorder.phase("task_resolution"):
            tasks = _make_tasks()
        stage = "evaluation"
        tracker = EvaluationTracker(output_path=str(artifact_root))
        results = _evaluate(tasks, recorder=recorder, tracker=tracker, cache_requests_enabled=case_id != "cache-disabled")
        samples = results.pop("samples")
        correctness = _correctness_projection(results, samples)
        stage = "artifact_stage"
        with recorder.phase("artifact_stage"):
            tracker.save_results_aggregated(results=results, samples=samples, datetime_str=stem)
            for task_name in sorted(samples):
                tracker.save_results_samples(task_name=task_name, samples=samples[task_name])
            artifact_files = sum(path.is_file() for path in artifact_root.rglob("*"))
            if artifact_files != 3:
                raise RuntimeError("artifact stage did not write exactly three files")
        recorder.increment("artifact_files", artifact_files)
        recorder.counters["batches"] = recorder.counters["inference_dispatches"]
        correctness["artifact_files"] = artifact_files
        recorder.set_resource("correctness_digest", _sha256(correctness))
        recorder.set_resource("outcome", "completed")
        recorder.finish()
        finished = True
        duration_s = recorder.resources["end_to_end_duration_ns"] / 1_000_000_000
        recorder.counters["end_to_end_scored_documents_per_second"] = recorder.counters["scored_documents"] / duration_s
        stage = "cache_validation"
        cache_after = _cache_snapshot()
        expected_cache = {"cache-disabled": (0, 0, True), "cache-cold": (0, 2, False), "cache-warm": (2, 2, True)}[case_id]
        if (len(cache_before), len(cache_after), cache_before == cache_after) != expected_cache:
            raise RuntimeError("request cache lifecycle did not match the benchmark case")
    except BaseException as exc:
        if recorder.counters["failures"] == 0:
            recorder.increment("failures")
        recorder.set_resource("failure", {"stage": stage, "type": type(exc).__name__})
        recorder.set_resource("outcome", "failed")
        recorder.set_resource("partial_artifacts", {"policy": "preserve", "files": sum(path.is_file() for path in artifact_root.rglob("*"))})
        try:
            if not started:
                recorder.start()
            if not finished:
                recorder.finish()
            recorder.write_json(failed_path)
        except BaseException:
            pass
        raise
    return recorder.write_json(record_path)


def _summarize(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        if not record["repetition"]["warmup"]:
            grouped.setdefault(record["repetition"]["case_id"], []).append(record)
    summary = {}
    for case_id, case_records in grouped.items():
        metrics = {
            "end_to_end_duration_ns": [record["resources"]["end_to_end_duration_ns"] for record in case_records],
            "peak_host_rss_bytes": [record["resources"]["peak_host_rss_bytes"] for record in case_records],
            "end_to_end_scored_documents_per_second": [record["counters"]["end_to_end_scored_documents_per_second"] for record in case_records],
        }
        for phase in {phase["name"] for record in case_records for phase in record["phases"]}:
            metrics[f"phase.{phase}.duration_ns"] = [next(item["duration_ns"] for item in record["phases"] if item["name"] == phase) for record in case_records]
        summary[case_id] = {
            name: {
                "median": statistics.median(values),
                "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                "pvariance": statistics.pvariance(values),
            }
            for name, values in metrics.items()
        }
    return summary


def run_suite(output_dir: Path, *, warmups: int = 1, measured_repetitions: int = 5, digest_legacy_arguments: bool = False) -> list[Path]:
    if warmups < 0 or measured_repetitions < 1:
        raise ValueError("warmups must be non-negative and measured_repetitions must be positive")
    output_dir = Path(output_dir)
    repetitions = [(index, index < warmups) for index in range(warmups + measured_repetitions)]
    for case_id in _CASES:
        for index, warmup in repetitions:
            label = "warmup" if warmup else "measured"
            artifact_root = output_dir / "artifacts" / f"{case_id}-{index:03d}-{label}"
            if artifact_root.exists():
                raise FileExistsError(artifact_root)
            for suffix in (".json", "-failed.json"):
                path = output_dir / "records" / f"{case_id}-{index:03d}-{label}{suffix}"
                if path.exists():
                    raise FileExistsError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(tempfile.mkdtemp(prefix="lmms-eval-benchmark-cache-"))
    original_cache_path = request_cache.PATH
    paths = []
    try:
        for case_id in _CASES:
            if case_id == "cache-warm":
                request_cache.PATH = str(cache_root / "warm")
                _evaluate(_make_tasks(), cache_requests_enabled=True)
            for index, warmup in repetitions:
                request_cache.PATH = str(cache_root / ("warm" if case_id == "cache-warm" else f"{case_id}-{index}"))
                paths.append(_run_repetition(output_dir, case_id, index, warmup, digest_legacy_arguments))
    finally:
        request_cache.PATH = original_cache_path
        shutil.rmtree(cache_root)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({record["resources"]["correctness_digest"] for record in records}) != 1:
        raise RuntimeError("benchmark correctness parity failed")
    print(json.dumps(_summarize(records), sort_keys=True))
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hermetic evaluator phase benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--measured-repetitions", type=int, default=5)
    parser.add_argument("--digest-legacy-arguments", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_suite(args.output_dir, warmups=args.warmups, measured_repetitions=args.measured_repetitions, digest_legacy_arguments=args.digest_legacy_arguments)


if __name__ == "__main__":
    main()

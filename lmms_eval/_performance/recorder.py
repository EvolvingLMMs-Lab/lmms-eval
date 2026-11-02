import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .json_v1 import canonical_json_bytes, validate_v1_json
from .legacy_invocation import build_legacy_invocation, validate_legacy_invocation
from .provenance import capture_baseline_provenance, peak_host_rss_bytes

PHASE_ORDER = ("queue_wait", "model_load", "task_resolution", "request_build", "preprocess", "inference", "filter_and_normalize", "score", "aggregate", "artifact_stage", "reset")
CACHE_STATES = {"cold", "warm", "mixed", "disabled"}
_PHASE_OWNERS = {"control", "worker", "publisher"}
_FORBIDDEN_IDENTITY_KEYS = {"eval_spec", "identity", "intent_id", "resolved_eval_id", "spec_digest", "runtime_id", "run_id", "attempt_id"}
_REPETITION_KEYS = {"suite_id", "case_id", "repetition_index", "warmup"}
_SAFE_INTEGER_MAX = 2**53 - 1
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _validate_repetition(repetition: Mapping[str, Any]) -> dict[str, Any]:
    if set(repetition) != _REPETITION_KEYS:
        raise ValueError(f"repetition must contain exactly {sorted(_REPETITION_KEYS)}")
    if type(repetition["suite_id"]) is not str or not repetition["suite_id"]:
        raise TypeError("repetition suite_id must be a non-empty string")
    if type(repetition["case_id"]) is not str or not repetition["case_id"]:
        raise TypeError("repetition case_id must be a non-empty string")
    if type(repetition["repetition_index"]) is not int or not 0 <= repetition["repetition_index"] <= _SAFE_INTEGER_MAX:
        raise TypeError("repetition repetition_index must be a non-negative IEEE-754 safe integer")
    if type(repetition["warmup"]) is not bool:
        raise TypeError("repetition warmup must be a boolean")
    return dict(repetition)


def _future_identity_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if found := key if key in _FORBIDDEN_IDENTITY_KEYS else _future_identity_key(item):
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            if found := _future_identity_key(item):
                return found
    return None


def _validate_record(record: Mapping[str, Any]) -> None:
    if type(record["schema_version"]) is not int or record["schema_version"] != 1 or record["record_kind"] != "baseline":
        raise ValueError("invalid baseline record header")
    if type(record["source_commit"]) is not str or not record["source_commit"]:
        raise TypeError("source_commit must be a non-empty string")
    for name in ("source_tree_digest", "environment_lock_digest"):
        if type(record[name]) is not str or _SHA256_RE.fullmatch(record[name]) is None:
            raise ValueError(f"{name} must be a SHA-256 digest")
    if type(record["hardware"]) is not str or not record["hardware"] or record["cache_state"] not in CACHE_STATES:
        raise ValueError("invalid hardware or cache state")
    _validate_repetition(record["repetition"])
    validate_legacy_invocation(record["legacy_invocation"])
    for phase in record["phases"]:
        if set(phase) != {"name", "owner", "duration_ns", "overlapped"}:
            raise ValueError("invalid phase fields")
        if phase["name"] not in PHASE_ORDER or phase["owner"] not in _PHASE_OWNERS:
            raise ValueError("invalid phase name or owner")
        if type(phase["duration_ns"]) is not int or not 0 <= phase["duration_ns"] <= _SAFE_INTEGER_MAX:
            raise TypeError("phase duration_ns must be a non-negative IEEE-754 safe integer")
        if type(phase["overlapped"]) is not bool:
            raise TypeError("phase overlapped must be a boolean")
    if type(record["counters"]) is not dict or type(record["resources"]) is not dict:
        raise TypeError("counters and resources must be open maps")
    validate_v1_json(record, path="$", allow_null=True)


@dataclass
class _AccumulatedPhase:
    duration_ns: int = 0
    owner: str = "worker"
    overlapped: bool = False


@dataclass
class BaselinePerformanceRecorder:
    source_commit: str
    source_tree_digest: str
    legacy_invocation: dict[str, Any]
    environment_lock_digest: str
    hardware: str
    cache_state: str
    repetition: dict[str, Any]
    counters: dict[str, Any] = field(default_factory=lambda: {"failures": 0})
    resources: dict[str, Any] = field(default_factory=dict)
    _phases: dict[str, _AccumulatedPhase] = field(default_factory=dict)
    _start_ns: int | None = None
    _finished: bool = False
    _active_phases: int = 0

    @classmethod
    def capture(cls, *, repo_root: Path, legacy_arguments: Mapping[str, Any], cache_state: str, repetition: Mapping[str, Any], digest_legacy_arguments: bool = False) -> "BaselinePerformanceRecorder":
        if cache_state not in CACHE_STATES:
            raise ValueError(f"invalid cache state: {cache_state}")
        provenance = capture_baseline_provenance(repo_root)
        invocation = build_legacy_invocation(legacy_arguments, digest_only=digest_legacy_arguments)
        return cls(provenance.source_commit, provenance.source_tree_digest, invocation, provenance.environment_lock_digest, provenance.hardware, cache_state, _validate_repetition(repetition))

    def start(self) -> None:
        if self._start_ns is not None:
            raise RuntimeError("recorder already started")
        self._start_ns = time.perf_counter_ns()

    @contextmanager
    def phase(self, name: str, *, owner: str = "worker", overlapped: bool = False):
        if name not in PHASE_ORDER or owner not in _PHASE_OWNERS:
            raise ValueError(f"invalid phase: {name}/{owner}")
        if type(overlapped) is not bool:
            raise TypeError("phase overlapped must be a boolean")
        if self._start_ns is None:
            raise RuntimeError("recorder has not started")
        if self._finished:
            raise RuntimeError("recorder is already finished")
        current = self._phases.get(name)
        if current is None:
            current = self._phases[name] = _AccumulatedPhase(owner=owner, overlapped=overlapped)
        elif (current.owner, current.overlapped) != (owner, overlapped):
            raise ValueError(f"inconsistent phase metadata: {name}")
        started = time.perf_counter_ns()
        self._active_phases += 1
        try:
            yield
        except BaseException:
            self.increment("failures")
            raise
        finally:
            try:
                current.duration_ns += time.perf_counter_ns() - started
            finally:
                self._active_phases -= 1

    def increment(self, name: str, amount: int | float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def set_resource(self, name: str, value: Any) -> None:
        self.resources[name] = value

    def finish(self) -> None:
        if self._start_ns is None:
            raise RuntimeError("recorder has not started")
        if self._finished:
            raise RuntimeError("recorder is already finished")
        if self._active_phases:
            raise RuntimeError("recorder has an active phase")
        self.resources["end_to_end_duration_ns"] = time.perf_counter_ns() - self._start_ns
        self.resources["peak_host_rss_bytes"] = peak_host_rss_bytes()
        self._finished = True

    def to_record(self) -> dict[str, Any]:
        if self._active_phases:
            raise RuntimeError("recorder has an active phase")
        if not self._finished:
            raise RuntimeError("recorder must be finished before record emission")
        unknown_phases = set(self._phases).difference(PHASE_ORDER)
        if unknown_phases:
            raise ValueError(f"invalid phase names: {sorted(unknown_phases)}")
        phases = [{"name": name, "owner": value.owner, "duration_ns": value.duration_ns, "overlapped": value.overlapped} for name in PHASE_ORDER if (value := self._phases.get(name)) is not None]
        record = {
            "schema_version": 1,
            "record_kind": "baseline",
            "source_commit": self.source_commit,
            "source_tree_digest": self.source_tree_digest,
            "legacy_invocation": self.legacy_invocation,
            "environment_lock_digest": self.environment_lock_digest,
            "hardware": self.hardware,
            "cache_state": self.cache_state,
            "repetition": self.repetition,
            "phases": phases,
            "counters": dict(self.counters),
            "resources": dict(self.resources),
        }
        if key := _future_identity_key(record):
            raise ValueError(f"future identity is not allowed in baseline record: {key}")
        _validate_record(record)
        return deepcopy(record)

    def write_json(self, path: Path) -> Path:
        payload = canonical_json_bytes(self.to_record())
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False)
        temporary_path = Path(temporary.name)
        try:
            with temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return path

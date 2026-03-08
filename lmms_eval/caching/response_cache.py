"""Unified response-level cache for lmms-eval.

SQLite primary store (WAL mode) + JSONL write-ahead audit log.
Caches only deterministic requests (temperature=0, do_sample=False).
Write order: JSONL append+fsync -> SQLite upsert (crash-safe).

Activation::

    python -m lmms_eval --model ... --tasks ... --use_cache ./my_cache.db

Cache key: sha256(request_type, task_name, doc_id, idx, canonical gen_kwargs, content_hash, task_fingerprint, model_fingerprint_hash).

File layout:
    Single GPU, local disk   - writes directly to the user-specified .db file.
    Multi-GPU, local disk    - each rank writes to a temporary shard
                               (``<target>.shard.<rank>``); rank 0 merges after eval.
    Remote target (NFS/CIFS) - two-tier mode: writes go to local scratch (NVMe/SSD),
                               reads check local first then shared DB on NFS.
                               After eval, local writes merge back to NFS target.
    Layered directory mode   - shared root DB lives at ``<cache_root>/cache.db`` while
                               each run writes to ``<cache_root>/runs/<run_id>/``.
                               Rank 0 merges ready runs back into the root DB under an
                               exclusive lock, so concurrent jobs do not fight over writes.
"""

import hashlib
import inspect
import json
import os
import sqlite3
import socket
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import Any, Dict, List, Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.instance import GenerationResult, Instance
from lmms_eval.caching.fs_detect import FsType, detect_fs_type, find_local_scratch

CACHE_RELEVANT_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "max_gen_toks",
        "do_sample",
        "num_beams",
        "until",
        "repetition_penalty",
        "n",
        "best_of",
        "num_return_sequences",
    }
)

_SCHEMA_VERSION = 3
_LAYERED_RUNS_DIRNAME = "runs"
_CACHE_RUN_ID_ENV_KEYS = ("LMMS_CACHE_RUN_ID", "SLURM_JOB_ID", "TORCHELASTIC_RUN_ID")
_LAYERED_READY_MARKER = ".ready"
_LAYERED_MERGED_MARKER = ".merged"
_LAYERED_LOCK_DIRNAME = ".merge.lock"


@dataclass(frozen=True)
class ResponseCacheLayout:
    mode: str
    target_db_path: str
    target_audit_path: str
    write_db_path: str
    write_audit_path: str
    shared_db_path: Optional[str]
    cleanup_after_consolidation: bool
    run_id: str = ""
    run_dir: str = ""
    run_root: str = ""
    cache_root: str = ""
    staging_dir: Optional[str] = None


def _short_hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _sanitize_run_id(run_id: str) -> str:
    safe = []
    for ch in str(run_id):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("-")
    sanitized = "".join(safe).strip("-._")
    return sanitized or "run"


def _resolve_cache_run_id(world_size: int) -> str:
    for env_key in _CACHE_RUN_ID_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if env_value:
            return _sanitize_run_id(env_value)

    if world_size > 1:
        master_addr = os.environ.get("MASTER_ADDR", "master")
        master_port = os.environ.get("MASTER_PORT", "0")
        return _sanitize_run_id(f"{master_addr}-{master_port}-ws{world_size}")

    return uuid.uuid4().hex


def _looks_like_cache_directory(path: str) -> bool:
    if not path:
        return False
    if path.endswith(os.sep):
        return True
    if os.path.isdir(path):
        return True
    if os.path.exists(path):
        return False
    return os.path.splitext(os.path.basename(path))[1].lower() != ".db"


def _looks_like_layered_cache_db(path: str) -> bool:
    if not path:
        return False
    return os.path.basename(path).lower() == "cache.db"


def _is_layered_cache_target(path: str) -> bool:
    return _looks_like_cache_directory(path) or _looks_like_layered_cache_db(path)


def _resolve_layered_target_paths(use_cache: str) -> tuple[str, str, str, str]:
    if _looks_like_cache_directory(use_cache):
        cache_root = os.path.abspath(use_cache.rstrip(os.sep))
        target_db_path = os.path.join(cache_root, "cache.db")
        target_audit_path = os.path.join(cache_root, "cache.audit.jsonl")
    elif _looks_like_layered_cache_db(use_cache):
        target_db_path = os.path.abspath(use_cache)
        target_audit_path = os.path.splitext(target_db_path)[0] + ".audit.jsonl"
        cache_root = os.path.dirname(target_db_path)
    else:
        target_db_path = use_cache
        if target_db_path.endswith(os.sep):
            target_db_path = os.path.join(target_db_path, "cache.db")
        elif not target_db_path.endswith(".db"):
            target_db_path = target_db_path + ".db"
        target_db_path = os.path.abspath(target_db_path)
        target_audit_path = os.path.splitext(target_db_path)[0] + ".audit.jsonl"
        cache_root = os.path.dirname(target_db_path)

    run_root = os.path.join(cache_root, _LAYERED_RUNS_DIRNAME)
    return cache_root, target_db_path, target_audit_path, run_root


def _touch_text(path: str, content: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


@contextmanager
def _merge_lock(lock_dir: str, timeout_seconds: int = 60, poll_interval_seconds: float = 1.0):
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            os.mkdir(lock_dir)
            owner_path = os.path.join(lock_dir, "owner.json")
            _touch_text(
                owner_path,
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "created_at": time.time(),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            )
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for cache merge lock: {lock_dir}")
            time.sleep(poll_interval_seconds)

    try:
        yield
    finally:
        owner_path = os.path.join(lock_dir, "owner.json")
        try:
            if os.path.exists(owner_path):
                os.remove(owner_path)
        except OSError:
            pass
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def resolve_response_cache_layout(
    use_cache: str,
    *,
    world_size: int,
    global_rank: int,
    model_hash: str,
) -> ResponseCacheLayout:
    """Resolve response-cache paths for file mode, two-tier mode, or layered dir mode."""
    if _is_layered_cache_target(use_cache):
        cache_root, target_db_path, target_audit_path, run_root = _resolve_layered_target_paths(use_cache)
        os.makedirs(cache_root, exist_ok=True)
        run_id = _resolve_cache_run_id(world_size)
        os.makedirs(run_root, exist_ok=True)
        run_dir = os.path.join(run_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        if world_size > 1:
            write_db_path = os.path.join(run_dir, f"cache.db.shard.{global_rank}")
            write_audit_path = os.path.join(run_dir, f"cache.db.audit.shard.{global_rank}.jsonl")
        else:
            write_db_path = os.path.join(run_dir, "cache.db")
            write_audit_path = os.path.join(run_dir, "cache.audit.jsonl")
        shared_db_path = target_db_path if os.path.exists(target_db_path) else None
        return ResponseCacheLayout(
            mode="layered_dir",
            target_db_path=target_db_path,
            target_audit_path=target_audit_path,
            write_db_path=write_db_path,
            write_audit_path=write_audit_path,
            shared_db_path=shared_db_path,
            cleanup_after_consolidation=False,
            run_id=run_id,
            run_dir=run_dir,
            run_root=run_root,
            cache_root=cache_root,
        )

    target_db_path = use_cache
    if target_db_path.endswith(os.sep):
        target_db_path = os.path.join(target_db_path, "cache.db")
    elif not target_db_path.endswith(".db"):
        target_db_path = target_db_path + ".db"
    target_db_path = os.path.abspath(target_db_path)
    target_audit_path = os.path.splitext(target_db_path)[0] + ".audit.jsonl"
    target_dir = os.path.dirname(target_db_path)
    os.makedirs(target_dir, exist_ok=True)

    target_fs = detect_fs_type(target_db_path)
    if target_fs == FsType.REMOTE:
        local_scratch = find_local_scratch()
        if local_scratch is not None:
            local_cache_dir = os.path.join(local_scratch, "lmms_eval_cache", model_hash)
            os.makedirs(local_cache_dir, exist_ok=True)
            shared_db_path = target_db_path if os.path.exists(target_db_path) else None
            if world_size > 1:
                write_db_path = os.path.join(local_cache_dir, f"shard.{global_rank}.db")
                write_audit_path = os.path.join(local_cache_dir, f"shard.{global_rank}.audit.jsonl")
                staging_dir = os.path.splitext(target_db_path)[0] + ".staging"
                os.makedirs(staging_dir, exist_ok=True)
            else:
                write_db_path = os.path.join(local_cache_dir, "local.db")
                write_audit_path = os.path.join(local_cache_dir, "local.audit.jsonl")
                staging_dir = None
            return ResponseCacheLayout(
                mode="two_tier",
                target_db_path=target_db_path,
                target_audit_path=target_audit_path,
                write_db_path=write_db_path,
                write_audit_path=write_audit_path,
                shared_db_path=shared_db_path,
                cleanup_after_consolidation=True,
                staging_dir=staging_dir,
            )
        eval_logger.warning("ResponseCache: target is on remote FS but no local scratch found, writing directly")

    if world_size > 1:
        write_db_path = f"{target_db_path}.shard.{global_rank}"
        write_audit_path = f"{target_db_path}.audit.shard.{global_rank}.jsonl"
    else:
        write_db_path = target_db_path
        write_audit_path = target_audit_path
    return ResponseCacheLayout(
        mode="direct",
        target_db_path=target_db_path,
        target_audit_path=target_audit_path,
        write_db_path=write_db_path,
        write_audit_path=write_audit_path,
        shared_db_path=None,
        cleanup_after_consolidation=True,
    )


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS responses (
    cache_key     TEXT PRIMARY KEY,
    request_type  TEXT NOT NULL,
    task_name     TEXT NOT NULL,
    doc_id        INTEGER NOT NULL,
    idx           INTEGER NOT NULL DEFAULT 0,
    gen_kwargs    TEXT,
    response      TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_doc ON responses(task_name, doc_id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def canonicalize_gen_kwargs(gen_kwargs: Optional[dict]) -> str:
    """Normalize gen_kwargs for consistent hashing.

    Only includes keys that affect generation output.
    Coerces ``0.0`` to ``0`` so float/int variants hash identically.
    """
    if not gen_kwargs:
        return "{}"
    filtered = {}
    for k in sorted(gen_kwargs.keys()):
        if k in CACHE_RELEVANT_KEYS:
            v = gen_kwargs[k]
            if isinstance(v, float) and v == int(v):
                v = int(v)
            filtered[k] = v
    return json.dumps(filtered, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def is_deterministic(request_type: str, gen_kwargs: Optional[dict]) -> bool:
    """Check if a request produces deterministic output (safe to cache).

    ``loglikelihood`` is always deterministic.  For generation requests,
    returns False when ``temperature > 0`` (many backends sample implicitly)
    or ``do_sample=True``, or when multiple return sequences are requested.
    """
    if request_type == "loglikelihood":
        return True

    if gen_kwargs is None:
        gen_kwargs = {}

    temp = gen_kwargs.get("temperature", 0)
    if temp is None:
        temp = 0

    if isinstance(temp, (int, float)) and temp > 0:
        return False

    do_sample = gen_kwargs.get("do_sample", False)
    if do_sample:
        return False

    n = gen_kwargs.get("n", 1) or 1
    best_of = gen_kwargs.get("best_of", 1) or 1
    num_ret = gen_kwargs.get("num_return_sequences", 1) or 1
    if max(n, best_of, num_ret) > 1:
        return False

    return True


def extract_gen_kwargs(instance: Instance) -> dict:
    """Extract gen_kwargs dict from Instance.args.

    Relies on the invariant that gen_kwargs is the **only** ``dict``
    in the args tuple across all simple/chat model layouts.
    """
    for arg in instance.args:
        if isinstance(arg, dict):
            return arg
    return {}


def fingerprint_callable(fn) -> str:
    """Best-effort stable fingerprint for a callable (function/partial/lambda).

    Returns ``module.qualname|source_hash`` when source is available,
    falls back to ``repr()`` for lambdas and built-ins.
    """
    if fn is None:
        return "none"

    parts: list[str] = []

    actual_fn = fn
    partial_args: tuple = ()
    partial_kwargs: dict = {}
    if isinstance(fn, partial):
        actual_fn = fn.func
        partial_args = fn.args
        partial_kwargs = fn.keywords or {}

    module = getattr(actual_fn, "__module__", "unknown")
    qualname = getattr(actual_fn, "__qualname__", getattr(actual_fn, "__name__", "unknown"))
    parts.append(f"{module}.{qualname}")

    try:
        source = inspect.getsource(actual_fn)
        parts.append(hashlib.sha256(source.encode("utf-8")).hexdigest()[:16])
    except (OSError, TypeError):
        parts.append(repr(actual_fn))

    if partial_args or partial_kwargs:
        try:
            partial_repr = json.dumps({"args": list(partial_args), "kwargs": partial_kwargs}, sort_keys=True, default=str)
            parts.append(partial_repr)
        except (TypeError, ValueError):
            parts.append(f"partial({len(partial_args)},{len(partial_kwargs)})")

    return "|".join(parts)


def _extract_content_hash(instance: Instance) -> str:
    """Hash leading text arguments to prevent cache-key collisions.

    Some flows can issue multiple deterministic requests that share the same
    ``(task_name, doc_id, idx, gen_kwargs)`` while differing in prompt text.
    This is common in multi-round / agentic generation loops.

    We hash the leading consecutive string arguments (for example context and
    continuation) so those requests do not alias to the same cache entry.
    """
    args = instance.args
    text_parts = []
    for arg in args:
        if isinstance(arg, str):
            text_parts.append(arg)
        else:
            break
    if not text_parts:
        return ""
    raw = "\0".join(text_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_cache_key(
    request_type: str,
    task_name: str,
    doc_id: Union[int, str],
    gen_kwargs: dict,
    idx: int = 0,
    task_fingerprint: str = "",
    content_hash: str = "",
    model_fingerprint_hash: str = "",
    eval_version: str = "",
) -> str:
    """Deterministic SHA-256 cache key for a model response.

    ``idx`` distinguishes multiple-choice options sharing the same ``doc_id``.
    ``content_hash`` distinguishes conditional vs unconditional loglikelihood
    requests that share the same (task_name, doc_id, idx).
    ``task_fingerprint`` enables automatic invalidation on YAML/prompt changes.
    ``model_fingerprint_hash`` ensures key-level model adapter isolation.
    ``eval_version`` isolates cache entries across lmms-eval releases.
    """
    payload = {
        "v": _SCHEMA_VERSION,
        "rt": request_type,
        "tn": task_name,
        "did": doc_id,
        "idx": idx,
        "gk": canonicalize_gen_kwargs(gen_kwargs),
    }
    if content_hash:
        payload["ch"] = content_hash
    if task_fingerprint:
        payload["tf"] = task_fingerprint
    if model_fingerprint_hash:
        payload["mfh"] = model_fingerprint_hash
    if eval_version:
        payload["ev"] = eval_version
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _serialize_response(response: Any) -> str:
    return json.dumps(response, ensure_ascii=False, default=str)


def _deserialize_response(stored: str) -> Any:
    """json.loads returns list for stored tuples — downstream tuple unpacking still works."""
    try:
        return json.loads(stored)
    except (json.JSONDecodeError, TypeError):
        return stored


class ResponseCache:
    """Unified response cache: SQLite (lookup) + JSONL (crash recovery).

    Supports an optional **shared read-only DB** for two-tier caching.
    When ``shared_db_path`` is provided, lookups check the local (writable)
    DB first, then fall back to the shared DB.  All writes go exclusively
    to the local DB.  This enables a pattern where the shared DB lives on
    NFS (slow writes, fast reads) while the local DB lives on NVMe.

    Write path: JSONL append+fsync -> SQLite upsert.
    On startup: replays JSONL tail into SQLite to recover incomplete writes.
    Skips caching for non-deterministic requests and error/empty responses.
    """

    def __init__(
        self,
        db_path: str,
        audit_path: str,
        model_fingerprint: str = "",
        task_fingerprints: Optional[Dict[str, str]] = None,
        shared_db_path: Optional[str] = None,
        eval_version: str = "",
    ):
        self.db_path = db_path
        self.audit_path = audit_path
        self.model_fingerprint = model_fingerprint
        self._model_fingerprint_hash = _short_hash(model_fingerprint)
        self._task_fingerprints: Dict[str, str] = task_fingerprints or {}
        self._eval_version = eval_version

        self.db = sqlite3.connect(db_path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA_SQL)

        if model_fingerprint:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("model_fingerprint", model_fingerprint))
        if self._model_fingerprint_hash:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("model_fingerprint_hash", self._model_fingerprint_hash))
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("schema_version", str(_SCHEMA_VERSION)))
        if eval_version:
            # Warn if DB was written by a different lmms-eval version
            row = self.db.execute("SELECT value FROM meta WHERE key = 'eval_version'").fetchone()
            if row and row[0] != eval_version:
                eval_logger.warning(f"ResponseCache: DB was last written by lmms-eval {row[0]}, " f"current version is {eval_version}. Cache keys now include version \u2014 " f"old entries will not match (safe, but no reuse).")
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("eval_version", eval_version))
        self.db.commit()

        # Optional shared (read-only) DB for two-tier caching.
        self._shared_db: Optional[sqlite3.Connection] = None
        self._shared_db_path = shared_db_path
        if shared_db_path and os.path.exists(shared_db_path):
            try:
                encoded_path = urllib.parse.quote(str(shared_db_path), safe="/")
                self._shared_db = sqlite3.connect(f"file:{encoded_path}?mode=ro&immutable=1", uri=True, timeout=10)
                shared_count = self._shared_db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
                eval_logger.info(f"ResponseCache: shared DB loaded ({shared_count} entries): {shared_db_path}")
            except Exception as e:
                eval_logger.warning(f"ResponseCache: failed to open shared DB {shared_db_path}: {e}")
                self._shared_db = None

        self._replay_audit_log()
        self._audit_file = open(audit_path, "a", encoding="utf-8")

        self._hits = 0
        self._hits_shared = 0
        self._misses = 0
        self._skipped = 0

    def _replay_audit_log(self) -> None:
        """Replay JSONL entries missing from SQLite (crash recovery)."""
        if not os.path.exists(self.audit_path):
            return

        replayed = 0
        try:
            with open(self.audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if not rec.get("deterministic", True) or not rec.get("cache_key"):
                            continue
                        cur = self.db.execute("SELECT 1 FROM responses WHERE cache_key = ?", (rec["cache_key"],))
                        if cur.fetchone() is None:
                            self.db.execute(
                                "INSERT INTO responses (cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (rec["cache_key"], rec["request_type"], rec["task_name"], rec["doc_id"], rec.get("idx", 0), rec.get("gen_kwargs", "{}"), rec["response"], rec.get("created_at", time.time())),
                            )
                            replayed += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            if replayed > 0:
                self.db.commit()
                eval_logger.info(f"ResponseCache: replayed {replayed} entries from audit log")
        except Exception as e:
            eval_logger.warning(f"ResponseCache: audit log replay failed: {e}")

    def _lookup(self, cache_key: str) -> Any:
        """Look up a cache key: local DB first, then shared DB."""
        # 1. Check local (writable) DB
        cur = self.db.execute("SELECT response FROM responses WHERE cache_key = ?", (cache_key,))
        row = cur.fetchone()
        if row is not None:
            return _deserialize_response(row[0])
        # 2. Fall back to shared (read-only) DB
        if self._shared_db is not None:
            try:
                cur = self._shared_db.execute("SELECT response FROM responses WHERE cache_key = ?", (cache_key,))
                row = cur.fetchone()
                if row is not None:
                    self._hits_shared += 1
                    return _deserialize_response(row[0])
            except Exception:
                pass  # shared DB failure is non-fatal
        return None

    def _log_to_audit(
        self,
        request_type: str,
        task_name: str,
        doc_id: Union[int, str],
        idx: int,
        gen_kwargs: dict,
        response: Any,
        *,
        cache_key: str = "",
        deterministic: bool = True,
        task_fingerprint: str = "",
        content_hash: str = "",
        model_fingerprint_hash: str = "",
    ) -> None:
        """Append every response to the JSONL audit log regardless of determinism.

        This provides real-time observability (``tail -f``) for ALL model responses,
        including non-deterministic ones that are never stored in SQLite.
        """
        now = time.time()
        record = {
            "cache_key": cache_key,
            "fingerprint_schema_version": _SCHEMA_VERSION,
            "request_type": request_type,
            "task_name": task_name,
            "doc_id": doc_id,
            "idx": idx,
            "gen_kwargs": canonicalize_gen_kwargs(gen_kwargs),
            "response": _serialize_response(response),
            "created_at": now,
            "deterministic": deterministic,
        }
        if task_fingerprint:
            record["task_fingerprint"] = task_fingerprint
        if content_hash:
            record["content_hash"] = content_hash
        if model_fingerprint_hash:
            record["model_fingerprint_hash"] = model_fingerprint_hash
        if self._eval_version:
            record["eval_version"] = self._eval_version
        self._audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._audit_file.flush()
        os.fsync(self._audit_file.fileno())

    def _store(self, cache_key: str, request_type: str, task_name: str, doc_id: Union[int, str], idx: int, gen_kwargs: dict, response: Any) -> None:
        """Store a deterministic response in the SQLite cache (JSONL logging is handled separately by ``_log_to_audit``)."""
        now = time.time()
        gen_kwargs_str = canonicalize_gen_kwargs(gen_kwargs)
        response_str = _serialize_response(response)

        self.db.execute(
            "INSERT OR REPLACE INTO responses (cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cache_key, request_type, task_name, doc_id, idx, gen_kwargs_str, response_str, now),
        )
        self.db.commit()

    @staticmethod
    def _extract_cacheable(response: Any) -> Any:
        """Extract the cache-safe payload from a model response.

        ``GenerationResult`` objects are reduced to their ``.text`` so that the
        cache stores only plain strings (token counts are ephemeral).
        """
        if isinstance(response, GenerationResult):
            return response.text
        return response

    @staticmethod
    def _is_valid_response(response: Any, request_type: str) -> bool:
        if response is None:
            return False
        if isinstance(response, GenerationResult):
            return bool(response.text and response.text.strip())
        if request_type == "loglikelihood":
            return isinstance(response, (list, tuple)) and len(response) == 2
        if isinstance(response, str) and response.strip() == "":
            return False
        return True

    def execute(self, lm: Any, reqtype: str, requests: List[Instance]) -> list:
        """Check cache -> run model on misses -> store results -> return merged list.

        Maintains positional ordering so the caller can ``zip(resps, requests)``.
        """
        if not requests:
            return []

        results: list = [None] * len(requests)
        uncached: List[Instance] = []
        uncached_indices: List[int] = []

        for i, req in enumerate(requests):
            gen_kwargs = extract_gen_kwargs(req)

            if not is_deterministic(reqtype, gen_kwargs):
                uncached.append(req)
                uncached_indices.append(i)
                self._skipped += 1
                continue

            ch = _extract_content_hash(req)
            tf = self._task_fingerprints.get(req.task_name, "")
            cache_key = compute_cache_key(
                request_type=reqtype,
                task_name=req.task_name,
                doc_id=req.doc_id,
                gen_kwargs=gen_kwargs,
                idx=req.idx,
                content_hash=ch,
                task_fingerprint=tf,
                model_fingerprint_hash=self._model_fingerprint_hash,
                eval_version=self._eval_version,
            )
            cached = self._lookup(cache_key)
            if cached is not None:
                results[i] = cached
                self._hits += 1
            else:
                uncached.append(req)
                uncached_indices.append(i)
                self._misses += 1

        n_hit_this_batch = len(requests) - len(uncached)
        if n_hit_this_batch > 0:
            eval_logger.info(f"ResponseCache: {n_hit_this_batch}/{len(requests)} cache hits ({self._skipped} non-deterministic skipped)")

        if uncached:
            new_resps = getattr(lm, reqtype)(uncached)
            for idx_pos, req, resp in zip(uncached_indices, uncached, new_resps):
                results[idx_pos] = resp
                cacheable = self._extract_cacheable(resp)
                gen_kwargs = extract_gen_kwargs(req)
                deterministic = is_deterministic(reqtype, gen_kwargs)
                ch = _extract_content_hash(req)
                tf = self._task_fingerprints.get(req.task_name, "")
                cache_key = (
                    compute_cache_key(
                        request_type=reqtype,
                        task_name=req.task_name,
                        doc_id=req.doc_id,
                        gen_kwargs=gen_kwargs,
                        idx=req.idx,
                        content_hash=ch,
                        task_fingerprint=tf,
                        model_fingerprint_hash=self._model_fingerprint_hash,
                        eval_version=self._eval_version,
                    )
                    if deterministic
                    else ""
                )
                self._log_to_audit(
                    reqtype,
                    req.task_name,
                    req.doc_id,
                    req.idx,
                    gen_kwargs,
                    cacheable,
                    cache_key=cache_key,
                    deterministic=deterministic,
                    task_fingerprint=tf,
                    content_hash=ch,
                    model_fingerprint_hash=self._model_fingerprint_hash,
                )
                if deterministic and self._is_valid_response(resp, reqtype):
                    self._store(cache_key, reqtype, req.task_name, req.doc_id, req.idx, gen_kwargs, cacheable)
        else:
            eval_logger.info(f"ResponseCache: all {len(requests)} requests served from cache — skipping model inference")

        return results

    def get_stats(self) -> Dict[str, Any]:
        total_lookups = self._hits + self._misses
        stats = {
            "hits": self._hits,
            "hits_shared": self._hits_shared,
            "misses": self._misses,
            "skipped_non_deterministic": self._skipped,
            "hit_rate": self._hits / max(1, total_lookups),
            "total_cached_entries": self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0],
        }
        if self._shared_db is not None:
            try:
                stats["shared_cached_entries"] = self._shared_db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            except Exception:
                stats["shared_cached_entries"] = "unavailable"
        return stats

    def close(self) -> None:
        try:
            if self._audit_file and not self._audit_file.closed:
                self._audit_file.close()
        except Exception:
            pass
        try:
            if self.db:
                self.db.close()
        except Exception:
            pass
        try:
            if self._shared_db:
                self._shared_db.close()
                self._shared_db = None
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def merge_shards(shard_paths: List[str], output_path: str) -> int:
        """Merge per-rank SQLite shards into a consolidated DB.

        Uses INSERT OR IGNORE so existing entries in ``output_path`` are preserved.
        Creates ``output_path`` if it does not exist.

        Returns the number of entries inserted.
        """
        out_db = sqlite3.connect(output_path, timeout=30)
        out_db.execute("PRAGMA journal_mode=WAL")
        out_db.executescript(_SCHEMA_SQL)

        total = 0
        for shard_path in shard_paths:
            if not os.path.exists(shard_path):
                continue
            shard_db = sqlite3.connect(shard_path)
            rows = shard_db.execute("SELECT cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response, created_at FROM responses").fetchall()
            for row in rows:
                try:
                    out_db.execute(
                        "INSERT OR IGNORE INTO responses (cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        row,
                    )
                    total += 1
                except sqlite3.IntegrityError:
                    pass
            # Copy meta table entries (model_fingerprint, schema_version, etc.)
            meta_rows = shard_db.execute("SELECT key, value FROM meta").fetchall()
            for key, value in meta_rows:
                out_db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
            shard_db.close()

        out_db.commit()
        out_db.close()
        return total

    @staticmethod
    def merge_audit_logs(audit_paths: List[str], output_path: str) -> int:
        """Merge per-rank JSONL audit logs into a single file.

        Appends entries from all ``audit_paths`` into ``output_path``,
        sorted by ``created_at`` timestamp.  Deduplicates by ``cache_key``
        for deterministic entries; non-deterministic entries are always kept.

        Returns the number of lines written.
        """
        entries: list = []
        seen_keys: set = set()
        for path in audit_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Deduplicate deterministic entries by cache_key
                    ck = rec.get("cache_key", "")
                    if ck and rec.get("deterministic", True):
                        if ck in seen_keys:
                            continue
                        seen_keys.add(ck)
                    entries.append(rec)

        # Sort by created_at for chronological ordering
        entries.sort(key=lambda r: r.get("created_at", 0))

        written = 0
        with open(output_path, "a", encoding="utf-8") as out:
            for rec in entries:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        return written

    @staticmethod
    def consolidate_cache(
        target_db_path: str,
        shard_db_paths: List[str],
        shard_audit_paths: List[str],
        target_audit_path: str,
        cleanup: bool = True,
    ) -> None:
        """Consolidate per-rank shards into a single cache DB + audit log.

        Called by rank 0 after evaluation completes.

        1. Merges all shard DBs into ``target_db_path`` (INSERT OR IGNORE).
        2. Merges all shard JSONL audit logs into ``target_audit_path``.
        3. If ``cleanup`` is True, removes the shard files.
        """
        # Merge SQLite shards
        merged_entries = ResponseCache.merge_shards(shard_db_paths, target_db_path)
        eval_logger.info(f"ResponseCache: consolidated {merged_entries} entries from " f"{len(shard_db_paths)} shard(s) into {target_db_path}")

        # Merge JSONL audit logs
        merged_lines = ResponseCache.merge_audit_logs(shard_audit_paths, target_audit_path)
        eval_logger.info(f"ResponseCache: consolidated {merged_lines} audit entries from " f"{len(shard_audit_paths)} log(s) into {target_audit_path}")

        # Cleanup shard files
        if cleanup:
            for path in shard_db_paths + shard_audit_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        # Also remove WAL/SHM sidecar files for SQLite
                        for suffix in ("-wal", "-shm"):
                            sidecar = path + suffix
                            if os.path.exists(sidecar):
                                os.remove(sidecar)
                except OSError as e:
                    eval_logger.warning(f"ResponseCache: failed to remove shard {path}: {e}")
            eval_logger.info("ResponseCache: shard files cleaned up")

    @staticmethod
    def mark_layered_run_ready(run_dir: str) -> None:
        _touch_text(os.path.join(run_dir, _LAYERED_READY_MARKER), f"{time.time():.6f}\n")

    @staticmethod
    def mark_layered_run_merged(run_dir: str) -> None:
        _touch_text(os.path.join(run_dir, _LAYERED_MERGED_MARKER), f"{time.time():.6f}\n")

    @staticmethod
    def _collect_layered_run_artifacts(run_dir: str) -> tuple[List[str], List[str]]:
        shard_db_paths = sorted(glob(os.path.join(run_dir, "cache.db.shard.*")))
        shard_audit_paths = sorted(glob(os.path.join(run_dir, "cache.db.audit.shard.*.jsonl")))

        single_db_path = os.path.join(run_dir, "cache.db")
        if not shard_db_paths and os.path.exists(single_db_path):
            shard_db_paths = [single_db_path]

        single_audit_path = os.path.join(run_dir, "cache.audit.jsonl")
        if not shard_audit_paths and os.path.exists(single_audit_path):
            shard_audit_paths = [single_audit_path]

        return shard_db_paths, shard_audit_paths

    @staticmethod
    def finalize_layered_runs(
        *,
        target_db_path: str,
        target_audit_path: str,
        run_root: str,
        current_run_dir: str,
        cleanup: bool = False,
        lock_timeout_seconds: int = 60,
    ) -> int:
        """Merge all completed layered runs for a shared cache target.

        Each run writes into its own UUID-scoped directory under ``run_root``.
        Rank 0 marks the current run as ready, then merges every ready-but-unmerged
        run under an exclusive lock so concurrent jobs do not fight over the root DB.
        """
        if not run_root or not current_run_dir:
            return 0

        os.makedirs(run_root, exist_ok=True)
        ResponseCache.mark_layered_run_ready(current_run_dir)
        lock_dir = os.path.join(run_root, _LAYERED_LOCK_DIRNAME)

        try:
            with _merge_lock(lock_dir, timeout_seconds=lock_timeout_seconds):
                merged_runs = 0
                for entry in sorted(os.scandir(run_root), key=lambda item: item.name):
                    if not entry.is_dir():
                        continue
                    run_dir = entry.path
                    ready_marker = os.path.join(run_dir, _LAYERED_READY_MARKER)
                    merged_marker = os.path.join(run_dir, _LAYERED_MERGED_MARKER)
                    if not os.path.exists(ready_marker) or os.path.exists(merged_marker):
                        continue

                    shard_db_paths, shard_audit_paths = ResponseCache._collect_layered_run_artifacts(run_dir)
                    if not shard_db_paths and not shard_audit_paths:
                        eval_logger.warning(f"ResponseCache: layered run has no cache artifacts, skipping merge: {run_dir}")
                        continue

                    ResponseCache.consolidate_cache(
                        target_db_path=target_db_path,
                        shard_db_paths=shard_db_paths,
                        shard_audit_paths=shard_audit_paths,
                        target_audit_path=target_audit_path,
                        cleanup=cleanup,
                    )
                    ResponseCache.mark_layered_run_merged(run_dir)
                    merged_runs += 1
                return merged_runs
        except TimeoutError as exc:
            eval_logger.warning(f"ResponseCache: layered merge deferred because lock is busy: {exc}")
            return 0

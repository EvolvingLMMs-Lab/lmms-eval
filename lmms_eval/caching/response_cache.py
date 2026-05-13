"""Unified response-level cache for lmms-eval.

SQLite primary store (WAL mode) + JSONL write-ahead audit log.
Caches only deterministic requests (temperature=0, do_sample=False).
Write order: JSONL append+fsync -> SQLite upsert (crash-safe).

Activation::

    python -m lmms_eval --model ... --tasks ... --use_cache /path/to/cache

Cache key: sha256(request_type, task_name, doc_id, idx, canonical gen_kwargs,
           content_hash, task_fingerprint, model_fingerprint_hash).

File layout::

    {cache_root}/
      cache.db                 # consolidated root DB
      cache.audit.jsonl
      runs/
        {run_id}/
          rank_{r}.db          # per-rank writes
          rank_{r}.audit.jsonl

Setup: ``ResponseCache.create(cache_root, model=..., model_args=..., task_dict=...)``
Teardown: ``cache.finalize(success=True, ...)``
"""

import hashlib
import inspect
import json
import os
import re
import shutil
import socket
import sqlite3
import time
import urllib.parse
import uuid
from contextlib import contextmanager
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
_CHECKPOINT_INTERVAL = 256  # responses between crash-safety checkpoints
_CHECKPOINT_INTERVAL_ENV = "LMMS_CACHE_CHECKPOINT_INTERVAL"

_FUNC_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")


def _get_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _get_env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


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

    @classmethod
    def create(
        cls,
        cache_root: str,
        *,
        model: str = "",
        model_args: Union[str, dict] = "",
        task_dict: Optional[dict] = None,
        world_size: int = 1,
        global_rank: int = 0,
    ) -> "ResponseCache":
        from lmms_eval.utils import (
            get_lmms_eval_cache_version,
            hash_string,
            simple_parse_args_string,
        )

        # Normalize cache_root
        if cache_root.endswith(".db"):
            cache_root = os.path.dirname(os.path.abspath(cache_root))
        cache_root = os.path.abspath(cache_root)

        # Task fingerprints
        task_fingerprints = {}
        if task_dict:
            for tname, tobj in task_dict.items():
                if hasattr(tobj, "dump_config"):
                    cfg_str = json.dumps(tobj.dump_config(), sort_keys=True, default=str)
                    cfg_str = _FUNC_ADDR_RE.sub(">", cfg_str)
                    task_fingerprints[tname] = hash_string(cfg_str)[:16]

        # Model fingerprint
        if isinstance(model_args, dict):
            model_args_fp = json.dumps(model_args, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
        elif isinstance(model_args, str):
            try:
                parsed = simple_parse_args_string(model_args)
            except Exception:
                parsed = model_args
            if isinstance(parsed, dict):
                model_args_fp = json.dumps(parsed, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
            else:
                model_args_fp = str(model_args)
        else:
            model_args_fp = str(model_args)
        model_fp = f"{model}|{model_args_fp}"
        model_hash = hash_string(model_fp)[:16]
        eval_version = get_lmms_eval_cache_version()

        # Create directories
        os.makedirs(cache_root, exist_ok=True)
        run_id = _resolve_cache_run_id(world_size)
        run_dir = os.path.join(cache_root, _LAYERED_RUNS_DIRNAME, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Determine write paths based on FS type
        target_db = os.path.join(cache_root, "cache.db")
        shared_db_path = target_db if os.path.exists(target_db) else None

        target_fs = detect_fs_type(cache_root)
        local_scratch = find_local_scratch() if target_fs == FsType.REMOTE else None

        rank_db_name = f"rank_{global_rank}.db"
        rank_audit_name = f"rank_{global_rank}.audit.jsonl"

        if local_scratch is not None:
            scratch_dir = os.path.join(local_scratch, "lmms_eval_cache", model_hash, "runs", run_id)
            os.makedirs(scratch_dir, exist_ok=True)
            write_db = os.path.join(scratch_dir, rank_db_name)
            write_audit = os.path.join(scratch_dir, rank_audit_name)
            use_scratch = True
        else:
            write_db = os.path.join(run_dir, rank_db_name)
            write_audit = os.path.join(run_dir, rank_audit_name)
            use_scratch = False

        eval_logger.info(f"ResponseCache: root={cache_root}, run={run_id}, rank={global_rank}/{world_size}, " f"writes={'scratch' if use_scratch else 'direct'}")

        instance = cls(
            db_path=write_db,
            audit_path=write_audit,
            model_fingerprint=model_fp,
            task_fingerprints=task_fingerprints,
            shared_db_path=shared_db_path,
            eval_version=eval_version,
        )
        # Store metadata for finalize()
        instance._cache_root = cache_root
        instance._run_id = run_id
        instance._run_dir = run_dir
        instance._global_rank = global_rank
        instance._world_size = world_size
        instance._use_scratch = use_scratch
        instance._checkpoint_interval = _get_env_int(_CHECKPOINT_INTERVAL_ENV, _CHECKPOINT_INTERVAL)
        instance._entries_since_checkpoint = 0
        if use_scratch:
            instance._remote_rank_db = os.path.join(run_dir, rank_db_name)
            instance._remote_rank_audit = os.path.join(run_dir, rank_audit_name)
        else:
            instance._remote_rank_db = None
            instance._remote_rank_audit = None
        return instance

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
        self.db: Optional[sqlite3.Connection] = None
        self._audit_file = None

        # Metadata set by create() for finalize()
        self._cache_root: Optional[str] = None
        self._run_id: str = ""
        self._run_dir: str = ""
        self._global_rank: int = 0
        self._world_size: int = 1
        self._use_scratch: bool = False
        self._remote_rank_db: Optional[str] = None
        self._remote_rank_audit: Optional[str] = None
        self._checkpoint_interval: int = _CHECKPOINT_INTERVAL
        self._entries_since_checkpoint: int = 0

        self._open_local_handles()

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

        self._hits = 0
        self._hits_shared = 0
        self._misses = 0
        self._skipped = 0

    def _open_local_handles(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        self.db = sqlite3.connect(self.db_path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA_SQL)

        if self.model_fingerprint:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("model_fingerprint", self.model_fingerprint))
        if self._model_fingerprint_hash:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("model_fingerprint_hash", self._model_fingerprint_hash))
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("schema_version", str(_SCHEMA_VERSION)))
        if self._eval_version:
            row = self.db.execute("SELECT value FROM meta WHERE key = 'eval_version'").fetchone()
            if row and row[0] != self._eval_version:
                eval_logger.warning(f"ResponseCache: DB was last written by lmms-eval {row[0]}, current version is {self._eval_version}. " f"Cache keys now include version — old entries will not match (safe, but no reuse).")
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("eval_version", self._eval_version))
        self.db.commit()
        self._replay_audit_log()
        self._audit_file = open(self.audit_path, "a", encoding="utf-8")

    def _close_local_handles(self, checkpoint: bool = False) -> None:
        try:
            if self._audit_file and not self._audit_file.closed:
                self._audit_file.flush()
                os.fsync(self._audit_file.fileno())
                self._audit_file.close()
        except Exception:
            pass
        self._audit_file = None

        try:
            if self.db:
                if checkpoint:
                    try:
                        self.db.commit()
                        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    except sqlite3.Error:
                        pass
                self.db.close()
        except Exception:
            pass
        self.db = None

    @staticmethod
    def _remove_sqlite_artifacts(db_path: str) -> None:
        for path in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

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
                    if self._use_scratch:
                        self._entries_since_checkpoint += 1
                        if self._entries_since_checkpoint >= self._checkpoint_interval:
                            self._checkpoint_to_run_dir()
        else:
            eval_logger.info(f"ResponseCache: all {len(requests)} requests served from cache — skipping model inference")

        return results

    def _checkpoint_to_run_dir(self) -> None:
        """Copy current scratch DB to the run directory for crash safety."""
        if not self._use_scratch or not self._remote_rank_db:
            return
        try:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.commit()
            shutil.copy2(self.db_path, self._remote_rank_db)
            if os.path.exists(self.audit_path):
                shutil.copy2(self.audit_path, self._remote_rank_audit)
            self._entries_since_checkpoint = 0
            eval_logger.debug(f"ResponseCache: checkpoint to {self._remote_rank_db}")
        except Exception as e:
            eval_logger.warning(f"ResponseCache: checkpoint failed: {e}")

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
        self._close_local_handles()
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

    def finalize(
        self,
        *,
        success: bool,
        dist_backend: str = "accelerate",
        accelerator: Any = None,
    ) -> None:
        """Log stats, close DB, barrier, and merge all rank shards into root cache.db.

        Call this once after evaluation completes. Handles:
        - Stats logging
        - Closing the DB
        - Copying scratch writes to shared run directory
        - Distributed barrier (all ranks must finish before merge)
        - Rank-0 merge of all rank DBs into cache_root/cache.db
        - Cleanup of the run directory after merge
        """
        # 1. Log stats
        try:
            stats = self.get_stats()
            shared_info = f", {stats.get('hits_shared', 0)} from shared DB" if stats.get("hits_shared", 0) else ""
            eval_logger.info(f"ResponseCache stats: {stats['hits']} hits{shared_info}, " f"{stats['misses']} misses, {stats['skipped_non_deterministic']} skipped, " f"hit rate: {stats['hit_rate']:.1%}")
        except Exception:
            pass

        # 2. Close DB (flushes WAL)
        self.close()

        if not success or self._cache_root is None:
            return

        # 3. Copy scratch to run dir
        if self._use_scratch and self._remote_rank_db:
            try:
                os.makedirs(self._run_dir, exist_ok=True)
                if os.path.exists(self.db_path):
                    shutil.copy2(self.db_path, self._remote_rank_db)
                if os.path.exists(self.audit_path) and self._remote_rank_audit:
                    shutil.copy2(self.audit_path, self._remote_rank_audit)
            except Exception as e:
                eval_logger.warning(f"ResponseCache: failed to copy scratch to run dir: {e}")

        # 4. Barrier
        self._distributed_barrier(dist_backend, accelerator)

        # 5. Rank 0: merge all rank DBs into root cache.db
        if self._global_rank == 0:
            self._merge_run_to_root()

    @staticmethod
    def _distributed_barrier(dist_backend: str, accelerator: Any) -> None:
        """Execute a distributed barrier."""
        try:
            if dist_backend == "accelerate" and accelerator is not None:
                accelerator.wait_for_everyone()
            elif dist_backend == "torchrun":
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
        except Exception as e:
            eval_logger.warning(f"ResponseCache: barrier failed: {e}")

    def _merge_run_to_root(self) -> None:
        """Rank-0 only: merge all rank DBs from the current run into cache_root/cache.db."""
        if not self._cache_root or not self._run_dir:
            return

        target_db = os.path.join(self._cache_root, "cache.db")
        target_audit = os.path.join(self._cache_root, "cache.audit.jsonl")
        run_root = os.path.join(self._cache_root, _LAYERED_RUNS_DIRNAME)
        lock_dir = os.path.join(run_root, _LAYERED_LOCK_DIRNAME)

        # Collect all rank DB/audit files from this run
        shard_dbs = sorted(glob(os.path.join(self._run_dir, "rank_*.db")))
        shard_audits = sorted(glob(os.path.join(self._run_dir, "rank_*.audit.jsonl")))

        # Also support legacy naming from older runs
        if not shard_dbs:
            shard_dbs = sorted(glob(os.path.join(self._run_dir, "cache.db.shard.*")))
            single_db = os.path.join(self._run_dir, "cache.db")
            if not shard_dbs and os.path.exists(single_db):
                shard_dbs = [single_db]
        if not shard_audits:
            shard_audits = sorted(glob(os.path.join(self._run_dir, "cache.db.audit.shard.*.jsonl")))
            single_audit = os.path.join(self._run_dir, "cache.audit.jsonl")
            if not shard_audits and os.path.exists(single_audit):
                shard_audits = [single_audit]

        if not shard_dbs and not shard_audits:
            eval_logger.info("ResponseCache: no rank artifacts to merge")
            return

        try:
            with _merge_lock(lock_dir, timeout_seconds=60):
                # Mark current run as ready
                _touch_text(os.path.join(self._run_dir, _LAYERED_READY_MARKER), f"{time.time():.6f}\n")

                # Merge current run
                if shard_dbs:
                    merged = ResponseCache.merge_shards(shard_dbs, target_db)
                    eval_logger.info(f"ResponseCache: merged {merged} entries from {len(shard_dbs)} rank(s) into {target_db}")
                if shard_audits:
                    merged_lines = ResponseCache.merge_audit_logs(shard_audits, target_audit)
                    eval_logger.info(f"ResponseCache: merged {merged_lines} audit entries into {target_audit}")

                # Mark merged
                _touch_text(os.path.join(self._run_dir, _LAYERED_MERGED_MARKER), f"{time.time():.6f}\n")

                # Opportunistically merge any other ready-but-unmerged runs
                self._merge_stale_runs(run_root, target_db, target_audit)

        except TimeoutError as exc:
            eval_logger.warning(f"ResponseCache: merge deferred, lock busy: {exc}")

    def _merge_stale_runs(self, run_root: str, target_db: str, target_audit: str) -> None:
        """Merge any previous runs that are ready but not yet merged."""
        for entry in sorted(os.scandir(run_root), key=lambda e: e.name):
            if not entry.is_dir() or entry.path == self._run_dir:
                continue
            ready = os.path.join(entry.path, _LAYERED_READY_MARKER)
            merged = os.path.join(entry.path, _LAYERED_MERGED_MARKER)
            if not os.path.exists(ready) or os.path.exists(merged):
                continue

            shard_dbs = sorted(glob(os.path.join(entry.path, "rank_*.db")))
            shard_audits = sorted(glob(os.path.join(entry.path, "rank_*.audit.jsonl")))
            # Legacy naming
            if not shard_dbs:
                shard_dbs = sorted(glob(os.path.join(entry.path, "cache.db.shard.*")))
                single = os.path.join(entry.path, "cache.db")
                if not shard_dbs and os.path.exists(single):
                    shard_dbs = [single]
            if not shard_audits:
                shard_audits = sorted(glob(os.path.join(entry.path, "cache.db.audit.shard.*.jsonl")))
                single = os.path.join(entry.path, "cache.audit.jsonl")
                if not shard_audits and os.path.exists(single):
                    shard_audits = [single]

            if shard_dbs:
                ResponseCache.merge_shards(shard_dbs, target_db)
            if shard_audits:
                ResponseCache.merge_audit_logs(shard_audits, target_audit)
            _touch_text(os.path.join(entry.path, _LAYERED_MERGED_MARKER), f"{time.time():.6f}\n")
            eval_logger.info(f"ResponseCache: merged stale run {entry.name}")

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

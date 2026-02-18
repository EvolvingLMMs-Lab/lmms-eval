"""Unified response-level cache for lmms-eval.

SQLite primary store (WAL mode) + JSONL write-ahead audit log.
Per-rank files for distributed safety. Caches only deterministic requests.
Write order: JSONL append+fsync -> SQLite upsert (crash-safe).

Activation: ``python -m lmms_eval --model ... --tasks ... --use_cache ./eval_cache``

Cache key: sha256(request_type, task_name, doc_id, idx, canonical gen_kwargs).
Scoped per model: ``{use_cache}/{model_hash}/rank{N}.db``
"""

import hashlib
import inspect
import json
import os
import sqlite3
import time
from functools import partial
from typing import Any, Dict, List, Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance

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

_SCHEMA_VERSION = 1

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
    """Hash the text content of loglikelihood args to prevent collisions.

    For multiple_choice with acc_mutual_info, conditional requests have
    ``(ctx, continuation, ...)`` while unconditional have ``("", choice)``.
    Both share the same (task_name, doc_id, idx) so we need this hash
    to distinguish them.
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
) -> str:
    """Deterministic SHA-256 cache key for a model response.

    ``idx`` distinguishes multiple-choice options sharing the same ``doc_id``.
    ``content_hash`` distinguishes conditional vs unconditional loglikelihood
    requests that share the same (task_name, doc_id, idx).
    ``task_fingerprint`` enables automatic invalidation on YAML/prompt changes.
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

    Write path: JSONL append+fsync -> SQLite upsert.
    On startup: replays JSONL tail into SQLite to recover incomplete writes.
    Skips caching for non-deterministic requests and error/empty responses.
    """

    def __init__(self, db_path: str, audit_path: str, model_fingerprint: str = "", task_fingerprints: Optional[Dict[str, str]] = None):
        self.db_path = db_path
        self.audit_path = audit_path
        self.model_fingerprint = model_fingerprint
        self._task_fingerprints: Dict[str, str] = task_fingerprints or {}

        self.db = sqlite3.connect(db_path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA_SQL)

        if model_fingerprint:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("model_fingerprint", model_fingerprint))
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("schema_version", str(_SCHEMA_VERSION)))
            self.db.commit()

        self._replay_audit_log()
        self._audit_file = open(audit_path, "a", encoding="utf-8")

        self._hits = 0
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
        cur = self.db.execute("SELECT response FROM responses WHERE cache_key = ?", (cache_key,))
        row = cur.fetchone()
        if row is None:
            return None
        return _deserialize_response(row[0])

    def _log_to_audit(self, request_type: str, task_name: str, doc_id: Union[int, str], idx: int, gen_kwargs: dict, response: Any, *, cache_key: str = "", deterministic: bool = True) -> None:
        """Append every response to the JSONL audit log regardless of determinism.

        This provides real-time observability (``tail -f``) for ALL model responses,
        including non-deterministic ones that are never stored in SQLite.
        """
        now = time.time()
        record = {
            "cache_key": cache_key,
            "request_type": request_type,
            "task_name": task_name,
            "doc_id": doc_id,
            "idx": idx,
            "gen_kwargs": canonicalize_gen_kwargs(gen_kwargs),
            "response": _serialize_response(response),
            "created_at": now,
            "deterministic": deterministic,
        }
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
    def _is_valid_response(response: Any, request_type: str) -> bool:
        """Reject None, empty strings, and malformed loglikelihood tuples to prevent cache poisoning."""
        if response is None:
            return False
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

            ch = _extract_content_hash(req) if reqtype == "loglikelihood" else ""
            tf = self._task_fingerprints.get(req.task_name, "")
            cache_key = compute_cache_key(
                request_type=reqtype,
                task_name=req.task_name,
                doc_id=req.doc_id,
                gen_kwargs=gen_kwargs,
                idx=req.idx,
                content_hash=ch,
                task_fingerprint=tf,
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
                gen_kwargs = extract_gen_kwargs(req)
                deterministic = is_deterministic(reqtype, gen_kwargs)
                ch = _extract_content_hash(req) if reqtype == "loglikelihood" else ""
                tf = self._task_fingerprints.get(req.task_name, "")
                cache_key = compute_cache_key(request_type=reqtype, task_name=req.task_name, doc_id=req.doc_id, gen_kwargs=gen_kwargs, idx=req.idx, content_hash=ch, task_fingerprint=tf) if deterministic else ""
                self._log_to_audit(reqtype, req.task_name, req.doc_id, req.idx, gen_kwargs, resp, cache_key=cache_key, deterministic=deterministic)
                if deterministic and self._is_valid_response(resp, reqtype):
                    self._store(cache_key, reqtype, req.task_name, req.doc_id, req.idx, gen_kwargs, resp)
        else:
            eval_logger.info(f"ResponseCache: all {len(requests)} requests served from cache — skipping model inference")

        return results

    def get_stats(self) -> Dict[str, Any]:
        total_lookups = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "skipped_non_deterministic": self._skipped,
            "hit_rate": self._hits / max(1, total_lookups),
            "total_cached_entries": self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0],
        }

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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def merge_shards(shard_paths: List[str], output_path: str) -> None:
        """Merge per-rank SQLite shards into a consolidated DB (INSERT OR IGNORE)."""
        if os.path.exists(output_path):
            os.remove(output_path)

        out_db = sqlite3.connect(output_path)
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
            shard_db.close()

        out_db.commit()
        out_db.close()
        eval_logger.info(f"ResponseCache: merged {total} entries from {len(shard_paths)} shards into {output_path}")

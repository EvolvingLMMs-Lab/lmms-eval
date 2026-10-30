## Request cache

`--cache_requests true` reuses document IDs and few-shot contexts. On a cache hit, the task constructs fresh requests with its current callbacks, generation settings, and metadata. Cached data never replaces live request objects or binds callbacks to an earlier task.

- Omit `--cache_requests` to disable both reads and writes.
- Use `--cache_requests refresh` to rebuild contexts without reading old cache files.
- Request cache files use bounded, hashed names that distinguish task classes and output types. Older Instance-based cache files are ignored and remain on disk.
- Cache publication replaces the old file only after the new data is fully serialized and written. A failed write preserves the previous file.

Request caching does not yet fingerprint every input to context construction. Use `refresh` when the dataset, prompt, few-shot seed, or context-building settings change. Request caches use pickle and must come from a trusted directory.

## Response Cache

`lmms-eval` includes a unified response cache backed by SQLite + JSONL write-ahead log. When enabled, deterministic model responses are stored and reused across runs, skipping redundant inference.

### Quick start

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks mme \
  --batch_size 1 \
  --use_cache ./eval_cache
```

On a second run with the same command, cached responses are loaded and the model is only called for new or changed requests.

When `--use_cache` points to a directory, or to an explicit root `cache.db`, lmms-eval uses a layered layout:

```text
eval_cache/
  cache.db
  cache.audit.jsonl
  runs/
    <run_id>/
      rank_0.db
      rank_0.audit.jsonl
```

The root `cache.db` is the shared read cache. Each evaluation run writes to its own UUID-scoped directory and rank 0 merges completed runs back into the root database under an exclusive lock. That gives you cache reuse without asking concurrent jobs to write into the same SQLite file.

### Shared filesystems and snapshots

SQLite WAL writers always use local storage. For NFS and other remote filesystems
(or an unrecognized filesystem), set `LMMS_LOCAL_CACHE` to a writable local
scratch directory. The automatic candidates are `/local/scratch`, `/scratch`,
and `/tmp`; candidates must be on a known local filesystem and have at least
1 GB free. If none qualify, cache creation fails rather than writing WAL on NFS.
Provision enough scratch space for rank databases, a full reader snapshot, and
the existing root database plus shards during a merge.

Rank checkpoints and the root `cache.db` are complete, closed SQLite files.
Rank 0 holds the merge lock, copies the previous root and shards to local storage,
merges there, then publishes with a temporary file, fsync, and atomic rename in
the destination directory. Readers copy one published file into private local
storage and use `immutable=1` only on that private copy. An existing reader keeps
its original view; a new run sees the latest publication. This keeps the existing
`cache.db` layout without accumulating versioned snapshots, at the cost of a full
copy per reader and merge. The filesystem must support atomic same-directory
rename and directory creation for the merge lock.

Checkpoint publication failures fail the run instead of marking incomplete data
ready. A ready run whose publication fails or whose merge lock times out remains
available for a later successful run to merge. A process killed while holding the
merge lock can leave `runs/.merge.lock`; inspect its `owner.json` and remove the
lock only after confirming the owning process is no longer running. Temporary
`.cache-*.tmp` files left by a killed publisher are not used as snapshots.

When upgrading, stop older cache writers before sharing the directory with this
implementation. Closed legacy databases remain readable. Databases with a
`-wal` or `-journal` sidecar are rejected: preserve the database and its sidecars,
recover/checkpoint them together on local storage while all writers are stopped,
and publish the closed database before reuse. Do not delete a WAL file to bypass
this check. `ResponseCache.merge_shards()` also expects closed inputs and a single
publisher; callers outside the evaluation lifecycle must serialize their merges.

To run the opt-in generation/cache integration test on an actual NFS mount:

```bash
LMMS_LOCAL_CACHE=/local/scratch \
LMMS_CACHE_E2E_ROOT=/mnt/nfs/cache-tests \
LMMS_CACHE_E2E_DEVICE=cuda \
python -m pytest test/cache/test_response_cache_integration.py -q -s
```

The test uses a tiny randomly initialized GPT-2, requires no model download, and
checks GPU generation followed by a run served entirely from the shared cache.
Use `LMMS_CACHE_E2E_DEVICE=cpu` when no GPU is available. The separate
`test/cache/test_response_cache_snapshots.py` suite checks concurrent publishers,
reader isolation, failed publication recovery, and simulated remote storage.

For concurrent publication and recovery on the real mount, also run:

```bash
LMMS_CACHE_E2E_ROOT=/mnt/nfs/cache-tests \
python -m pytest test/cache/test_response_cache_nfs.py -q
```

This suite skips unless filesystem detection identifies remote storage. It runs
four concurrent publishers with a pinned reader, injects a publication failure,
and checks recovery from a crashed process using a different local scratch
directory. It also asserts that every SQLite connection is local and that the
remote tree contains no SQLite journals. A loopback NFS mount exercises the NFS
protocol, but does not validate cache coherence between different client hosts.

### What gets cached

Only **deterministic** requests are cached. A request is considered non-deterministic (and skipped) when any of:

- `temperature > 0`
- `do_sample = True`
- `n > 1`, `best_of > 1`, or `num_return_sequences > 1`

`loglikelihood` requests are always deterministic.

Non-deterministic requests always go to the model, are never stored, and never returned from cache. This ensures `repeat > 1` with `temperature > 0` produces distinct results per repeat.

### Cache key

Each cached response is keyed by:

```
sha256({
    "v":   <schema_version>,        # auto-invalidates on schema upgrade
    "rt":  <request_type>,          # "generate_until" | "loglikelihood"
    "tn":  <task_name>,             # e.g. "mme"
    "did": <doc_id>,                # dataset sample ID
    "idx": <idx>,                   # multiple-choice option index within a doc
    "gk":  <canonicalized_gen_kwargs>,
    "ch":  <content_hash>,          # loglikelihood only: conditional vs unconditional
    "tf":  <task_fingerprint>       # sha256 of task YAML config
})
```

Only generation parameters that affect output are included in `gk`:

```
temperature, top_p, top_k, max_new_tokens, max_gen_toks,
do_sample, num_beams, until, repetition_penalty,
n, best_of, num_return_sequences
```

Float/int normalization: `temperature=0.0` and `temperature=0` produce the same key.

### File layout

Layered directory mode (recommended for shared or long-running jobs):

```
{cache_root}/
  cache.db
  cache.audit.jsonl
  runs/
    {run_id}/
      cache.db                     # single-rank writes
      cache.audit.jsonl
      cache.db.shard.{rank}        # multi-rank writes
      cache.db.audit.shard.{rank}.jsonl
      .ready
      .merged
```

Legacy file mode keeps the older behavior where a direct `.db` target may receive per-rank shard files next to the target DB.

### Cache invalidation

| Change | Effect |
|--------|--------|
| Different model or model_args | New `model_hash` directory |
| Edit task YAML or prompt function | New `task_fingerprint` in key |
| Change gen_kwargs (e.g. max_new_tokens) | Different `gk` in key |
| Schema version bump | Different `v` in key |

To force re-evaluation: delete the `{model_hash}/` directory under your cache path.

### Crash recovery

Write order: JSONL append + fsync -> SQLite upsert. On startup, valid JSONL entries missing from SQLite are replayed. Valid audit entries also replace invalid SQLite entries left by older versions. This survives crashes between the two writes.

### Poisoning prevention

Responses are validated before caching:

- `None` -> rejected
- Empty or whitespace-only strings -> rejected
- Loglikelihood responses that are not a two-element list or tuple -> rejected; element types are not validated

The same validity policy applies to audit recovery, local/shared cache reads, and shard merging. Invalid responses remain in the audit log with an empty cache key. Older keyed audit entries are validated before replay. Invalid SQLite payloads are treated as misses, and valid retries can replace them during replay or shard merging. Existing valid entries are preserved.

### Merge distributed shards

Layered directory mode merges distributed shards automatically on successful completion. Rank 0 acquires an exclusive merge lock, folds every ready run under `runs/` into the root `cache.db`, and marks the run directory as merged.

If you are using legacy file mode, you can still merge shard DBs manually:

```python
from lmms_eval.caching.response_cache import ResponseCache

ResponseCache.merge_shards(
    shard_paths=["eval_cache/cache.db.shard.0", "eval_cache/cache.db.shard.1"],
    output_path="eval_cache/cache.db",
)
```

### JSONL audit log

The JSONL file logs **all** model responses regardless of determinism. Each line includes a `"deterministic"` field. This provides real-time observability (`tail -f rank0.jsonl`) while only deterministic responses are stored in SQLite for cache reuse.

### Implementation

Source: `lmms_eval/caching/response_cache.py`
Tests: `test/cache/test_response_cache.py` (34 tests)

Covered: determinism detection, cache key collision, gen_kwargs extraction, poisoning prevention, hit/miss, non-deterministic bypass with repeats, JSONL audit log observability, crash recovery via JSONL replay, multi-rank isolation and shard merging, model fingerprint isolation, stats accuracy across close/reopen, large batch (1000 requests).

Not covered: `loglikelihood` end-to-end execute flow.

### What gets cached

- **Scope**: Per model instance and per task.
- **Unit**: One record per document (`doc_id`) with the final string response.
- **Files**: One JSONL file per task and process shard.

The cache is implemented in `lmms_eval.api.model.lmms` via:
- `load_cache()` and `load_jsonl_cache()` to load cached responses at startup
- `get_response_from_cache()` to split incoming requests into “already cached” vs “not cached”
- `add_request_response_to_cache()` to append new results as they are produced

Models that call these APIs (for example `async_openai`) automatically benefit from caching without any code changes in user scripts. You will need to use this api in your `generate_until` to cache and reload cache.

### Minimal example (inside a model's `generate_until`)

```python
def generate_until(self, requests):
    self.load_cache()
    cached, pending = self.get_response_from_cache(requests)
    results = [c["response"] for c in cached]
    for req in pending:
        out = call_backend(req)  # your model inference
        self.add_request_response_to_cache(req, out)
        results.append(out)
    return results
```

### Enable the cache

Set an environment variable before running:

```bash
export LMMS_EVAL_USE_CACHE=True
# optional: set the base directory for caches (defaults to ~/.cache/lmms-eval)
export LMMS_EVAL_HOME="/path/to/cache_root"
```

Nothing else is required. When enabled, the model will:
1) load existing JSONL cache files at startup; 2) serve responses from cache; 3) append newly generated responses back to the JSONL files.

### Where cache files live

- Base directory: `$(LMMS_EVAL_HOME:-~/.cache/lmms-eval)/eval_cache/<model_hash>/`
- File name per task and process shard: `{task_name}_rank{rank}_world_size{world_size}.jsonl`
- Record format per line:

```json
{"doc_id": <doc_id>, "response": <string>}
```

Notes:
- The `<model_hash>` is derived from a best‑effort human‑readable model identity (e.g., `model_version`) and the set of task names attached to the model, to avoid collisions.
- Separate files per `rank` and `world_size` make distributed runs safe to cache concurrently.

### How it works at runtime

For models wired to the cache API (e.g., `async_openai`):
- At the beginning of `generate_until(...)` the model calls `load_cache()` and then `get_response_from_cache(requests)`.
- Cached items are returned immediately; only the remaining requests are forwarded to the backend.
- After each response is produced, `add_request_response_to_cache(...)` appends a JSONL record.

The cache key is the tuple `(task_name, doc_id)`. Ensure your task produces stable `doc_id`s across runs.

### Example: use with async_openai

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"          # if your server allows it
export LMMS_EVAL_USE_CACHE=True         # enable JSONL cache
# optional: export LMMS_EVAL_HOME to relocate cache root

python -m lmms_eval \
  --model async_openai \
  --model_args model_version=grok-2-latest,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY \
  --tasks <your_task> \
  --batch_size 1 \
  --output_path ./logs/
```

On a second run with the same task/docs, cached responses will be loaded and only missing documents will call the model.

### Inspect or clear the cache

- Inspect: open the task JSONL file(s) under the model’s cache directory and view records.
- Clear: delete the corresponding JSONL file(s) or the entire `<model_hash>` directory to force re‑evaluation.

### Notes and limitations

- The JSONL cache is keyed by `task_name` and `doc_id`. Changing task names or document IDs invalidates reuse.
- Responses are cached as final strings. If your model emits intermediate tool calls, the final message (including any inline annotations) is what gets cached.
- Distributed runs write to per‑rank files to avoid contention; reusing the cache works across single‑ and multi‑GPU as long as `task_name`/`doc_id` match.

### Optional: legacy SQLite cache wrapper

There is also a separate optional wrapper `CachingLMM` (see `lmms_eval.api.model.CachingLMM`) that caches by hashing the entire call arguments to a SQLite DB (via `SqliteDict`). It is independent from the JSONL cache above and can be useful for broader API‑level caching. For most users, enabling `LMMS_EVAL_USE_CACHE=True` is sufficient and simpler.

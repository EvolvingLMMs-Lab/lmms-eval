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

```
{use_cache}/
  {model_hash}/          # sha256("{model}|{model_args}")[:16]
    rank0.db             # SQLite (WAL mode) - primary lookup
    rank0.jsonl          # write-ahead audit log - crash recovery
    rank1.db             # (if multi-GPU)
    rank1.jsonl
```

Per-rank files avoid write contention in distributed runs.

### Cache invalidation

| Change | Effect |
|--------|--------|
| Different model or model_args | New `model_hash` directory |
| Edit task YAML or prompt function | New `task_fingerprint` in key |
| Change gen_kwargs (e.g. max_new_tokens) | Different `gk` in key |
| Schema version bump | Different `v` in key |

To force re-evaluation: delete the `{model_hash}/` directory under your cache path.

### Crash recovery

Write order: JSONL append + fsync -> SQLite upsert. On startup, any JSONL entries missing from SQLite are replayed. This survives crashes between the two writes.

### Poisoning prevention

Responses are validated before caching:

- `None` -> rejected
- Empty or whitespace-only strings -> rejected
- Malformed loglikelihood tuples (not `[float, bool]`) -> rejected

### Merge distributed shards

After a multi-GPU run, merge per-rank DBs into one:

```python
from lmms_eval.caching.response_cache import ResponseCache

ResponseCache.merge_shards(
    shard_paths=["eval_cache/abc123/rank0.db", "eval_cache/abc123/rank1.db"],
    output_path="eval_cache/abc123/merged.db",
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

Models that call these APIs (for example `async_openai_compatible_chat`) automatically benefit from caching without any code changes in user scripts. You will need to use this api in your `generate_until` to cache and reload cache.

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

For models wired to the cache API (e.g., `async_openai_compatible_chat`):
- At the beginning of `generate_until(...)` the model calls `load_cache()` and then `get_response_from_cache(requests)`.
- Cached items are returned immediately; only the remaining requests are forwarded to the backend.
- After each response is produced, `add_request_response_to_cache(...)` appends a JSONL record.

The cache key is the tuple `(task_name, doc_id)`. Ensure your task produces stable `doc_id`s across runs.

### Example: use with async_openai_compatible_chat

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



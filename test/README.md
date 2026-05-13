# Test Suite

This directory contains the full test suite for lmms-eval. The suite validates every layer of the evaluation pipeline — from CLI argument parsing down to per-sample token counting — without requiring a GPU, dataset downloads, or API keys.

**292 tests pass in ~10 seconds on a CPU-only machine.**

## Quick Start

Run the entire suite (excluding API-dependent tests):

```bash
uv run python -m pytest test/ -q --ignore=test/eval/test_usage_metrics.py
```

Run a specific test file:

```bash
uv run python -m pytest test/eval/test_protocol.py -q
```

Run tests matching a keyword:

```bash
uv run python -m pytest test/ -k "cache and not agentic" -q
```

## Architecture

The tests mirror the evaluation pipeline. Each layer has dedicated coverage, and failures at any layer indicate a specific category of breakage.

```
User input: --model X --tasks Y
         │
         ▼
    ┌─ CLI Routing ──────────── test/cli/
    │   Parse arguments, dispatch to subcommands
    │
    ├─ Model Resolution ─────── test/models/
    │   Resolve model name -> class path, validate is_simple
    │
    ├─ Task Loading ─────────── test/eval/test_task_pipeline.py
    │   Discover YAML, verify registration, import utils    test/eval/test_benchmark_aliases.py
    │                                                       test/eval/test_new_benchmarks_tasks.py
    │
    ├─ Prompt Generation ────── test/eval/prompt_stability/
    │   Golden snapshot comparison for 8 classic benchmarks
    │
    ├─ Request Construction ─── test/eval/test_construct_requests.py
    │   Build Instance.args tuples for each task×model combination
    │
    ├─ Message Protocol ─────── test/eval/test_protocol.py
    │   ChatMessages structure, media extraction, HF conversion
    │
    ├─ Execution + Caching ──── test/eval/test_evaluator.py
    │   Agentic multi-round loop                            test/cache/test_response_cache.py
    │   Response cache lifecycle                            test/cache/test_agentic_response_cache.py
    │
    ├─ Token Counting ───────── test/eval/test_token_counts.py
    │   TokenCounts, GenerationResult, unwrap helpers
    │
    └─ Metric Aggregation ───── test/eval/test_efficiency_metrics.py
        Tokens-per-correct, budget tracking                 test/eval/test_usage_metrics.py (API)
```

## Test Modules in Detail

### CLI Layer

The CLI layer tests verify that user commands are parsed and routed correctly. For example, `--model openai --tasks mme` is detected as a legacy invocation and redirected to the `eval` subcommand:

```python
@pytest.mark.parametrize("argv, expected", [
    ([], False),
    (["--model", "openai", "--tasks", "mme"], True),  # legacy flags
    (["eval", "--model", "openai"], False),             # modern subcommand
])
def test_is_legacy_invocation(argv, expected):
    assert _is_legacy_invocation(argv) is expected
```

#### `test/cli/test_cli_dispatch_parametrized.py`

Parametrized test suite for the unified CLI dispatch router. Handles both legacy flag-style invocations (`--model X --tasks Y`) and modern subcommand-style invocations (`eval --model X`).

The test covers:

- **`_is_legacy_invocation()`** — Detects whether the user passed old-style flags so the router can redirect them to the `eval` subcommand for backward compatibility.
- **`_is_eval_wizard()`** — Detects the bare `eval` command (no flags) to launch interactive wizard mode.
- **`main()` routing** — Verifies that no-args prints a banner, `--help` prints usage, and subcommands dispatch correctly.
- **Subcommand parsers** — `tasks` accepts `list|groups|subtasks|tags`, `models` supports `--aliases`, `version` prints environment info.
- **`_col()` helper** — The column-formatting function used by `models` output. Short text is padded, long text is truncated.

Each test case is a single `@pytest.mark.parametrize` row instead of a separate method. This file is easier to extend when adding new CLI flags or subcommands.

#### `test/eval/test_cli_parse_args.py`

Focused test for the `--max_tokens` argument added to the evaluation CLI. Verifies that:

- Passing `--max_tokens 12345` sets `args.max_tokens = 12345`.
- Omitting the flag leaves `args.max_tokens` as `None`.

This pattern (one test file per new CLI argument) keeps argument parsing tests isolated and easy to find.

---

### Model Resolution Layer

The model resolution tests verify that user-provided model names map to the correct Python class paths. For example, when both chat and simple backends are registered, the registry defaults to chat:

```python
def test_chat_precedence_and_force_simple():
    registry = ModelRegistryV2()
    registry.register_manifest(ModelManifest(
        model_id="demo",
        simple_class_path="pkg.simple.DemoSimple",
        chat_class_path="pkg.chat.DemoChat",
    ))

    resolved = registry.resolve("demo")
    assert resolved.model_type == "chat"               # chat wins by default
    assert resolved.class_path == "pkg.chat.DemoChat"

    resolved = registry.resolve("demo", force_simple=True)
    assert resolved.model_type == "simple"              # explicit override
```

#### `test/models/test_model_registry_v2.py`

Validates `ModelRegistryV2`, the system that maps user-provided model names (like `qwen2_5_vl` or `openai_compatible`) to Python class paths.

The test covers:

- **Chat precedence** — When both `chat_class_path` and `simple_class_path` are registered, the registry defaults to chat. This matches the recommendation that all new models use the chat interface.
- **`force_simple=True`** — Forces resolution to the simple backend. If no simple class exists, the flag is silently ignored (falls back to chat).
- **Alias resolution** — `vllm_chat` resolves to `vllm`, `openai_compatible` resolves to `openai`. The `requested_name` field preserves what the user originally typed.
- **`_validate_model_class()`** — Type safety guard that runs after the class is imported. A chat-resolved model must have `is_simple = False`; a simple-resolved model must have `is_simple = True`. Mismatches raise `TypeError` with a clear message explaining the conflict.

These tests use synthetic `ModelManifest` registrations rather than the real registry to stay fast and isolated.

---

### Task Loading Layer

The task loading tests verify that task YAMLs parse correctly, are registered in the TaskManager, and their `utils.py` functions are importable. For example, each mainstream task must exist and use `generate_until`:

```python
MAINSTREAM_TASKS = ["mme", "mmmu_val", "mmstar", "ai2d", "scienceqa", "ocrbench", "mmvet", "videomme"]

@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_task_registered(task_name, tm):
    assert task_name in tm.all_subtasks

@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_task_output_type_is_generate_until(task_name, tm):
    yaml_path = _find_yaml(task_name, tm)
    config = yaml.safe_load(open(yaml_path))
    assert config.get("output_type") == "generate_until"
```

#### `test/eval/test_task_pipeline.py`

The most comprehensive task validation file. It checks eight mainstream tasks (mme, mmmu_val, mmstar, ai2d, scienceqa, ocrbench, mmvet, videomme) across four dimensions:

1. **Registration** — Each task name exists in `TaskManager.all_subtasks`. Each group name (mme, mmmu, mmbench, mathvista) exists in `TaskManager.all_tasks`. The total subtask count exceeds 100.
2. **YAML integrity** — Every task YAML file parses without error, contains a `task` field, has either `dataset_path` or `include`, and specifies an input formatter (`doc_to_messages`, `doc_to_text`, or `doc_to_visual`).
3. **Utils importability** — Each task's `utils.py` module can be imported. The `process_results` and `doc_to_text` functions exist and are callable. Tasks that use `doc_to_messages` (like mmmu_val) have that function too.
4. **Cross-task consistency** — All mainstream tasks use `generate_until` as their `output_type`. No duplicate task names exist in the registry.

> The `TaskManager` instantiation is expensive (~2s), so it is cached via a `@pytest.fixture(scope="module")` fixture and reused across all tests in the file.

#### `test/eval/test_benchmark_aliases.py`

Verifies that shorthand alias groups are registered and point to the correct underlying tasks:

| Alias | Target |
|-------|--------|
| `anet_qa` | `activitynetqa` |
| `mmmu_a` | `mmmu_val` |
| `egosch_a` | `egoschema` |

The test reads the alias YAML file and checks that the target task name appears in the `task` list.

#### `test/eval/test_new_benchmarks_tasks.py`

Covers recently added benchmarks (repcount, countix, ovr_kinetics, ssv2, vggsound, av_asr, neptune):

- **Registration** — All seven task names appear in the TaskManager.
- **`process_results` correctness** — Each task's result function is called with synthetic `doc` and model output, and the returned metrics (MAE, OBO, accuracy, WER) are checked against hand-computed expected values.
- **Dataset path verification** — Each task YAML has `dataset_path` pointing to the expected HuggingFace dataset (e.g., `lmms-lab-eval/repcount`), and none still reference local `data_files`.

---

### Prompt Stability Layer

Prompt stability tests catch silent drift in prompt templates. Each test calls the real `doc_to_text` with a synthetic document and compares against a golden snapshot:

```python
@pytest.mark.parametrize("case_name", sorted(CASES.keys()))
def test_prompt_stable(case_name, update_snapshots):
    case = CASES[case_name]
    doc_to_text = case["get_fn"]()
    prompt = doc_to_text(case["fixture"], case["default_kwargs"])

    snapshot_path = SNAPSHOT_DIR / f"{case_name}.json"
    expected = json.loads(snapshot_path.read_text())
    assert prompt == expected["prompt_text"], (
        f"Prompt changed for '{case_name}'!\n"
        f"If intentional, run: pytest test/eval/prompt_stability/ --update-snapshots"
    )
```

#### `test/eval/prompt_stability/`

This directory implements a snapshot-based regression guard for prompt templates. The core idea: if anyone changes a task YAML or its `doc_to_text` function, and that change alters what the model actually sees, these tests catch it immediately.

**How it works:**

1. Each test constructs a synthetic `doc` (minimal fake dataset row) for a specific benchmark.
2. It calls the **real** `doc_to_text` function from the task's `utils.py` with that doc.
3. It compares the resulting prompt string against a golden snapshot stored in `snapshots/<task>.json`.
4. If they differ, the test fails with a clear diff showing exactly what changed.

**Covered benchmarks (11 variants across 8 benchmarks):**

| Benchmark | Variant | Snapshot File |
|-----------|---------|---------------|
| MMMU | Multiple choice | `mmmu_val__mc.json` |
| MMMU | Open-ended | `mmmu_val__open.json` |
| MMBench | With hint | `mmbench_en_dev__hint.json` |
| MMBench | Without hint | `mmbench_en_dev__nohint.json` |
| HallusionBench | — | `hallusion_bench_image.json` |
| OCRBench | — | `ocrbench.json` |
| MMLongBench-Doc | — | `mmlongbench_doc.json` |
| VSI-Bench | Multiple choice | `vsibench__mca.json` |
| VSI-Bench | Numerical answer | `vsibench__na.json` |
| MMVP | — | `mmvp.json` |
| BLINK | Art style | `blink__art_style.json` |

Each benchmark also has a `gen_kwargs` test that verifies generation parameters (`until` stop sequences, `max_new_tokens`, etc.) match expectations.

**Updating snapshots after intentional changes:**

```bash
uv run python -m pytest test/eval/prompt_stability/ --update-snapshots
```

This regenerates all golden files. Review the diff before committing to confirm the prompt changes are intentional.

---

### Request Construction Layer

Request construction tests verify the exact tuple shape that models unpack from `Instance.args`. A mismatch here causes silent data corruption at runtime:

```python
def test_configurable_task_generate_until_tuple_shape():
    args = ("What is this?", {"temperature": 0}, _dummy_doc_to_visual, 0, "test_task", "test")
    inst = _make_instance("generate_until", args)

    prompt, gen_kwargs, doc_to_visual, doc_id, task, split = inst.args
    assert prompt == "What is this?"
    assert callable(doc_to_visual)
    assert isinstance(gen_kwargs, dict)
    assert doc_id == 0
```

#### `test/eval/test_construct_requests.py`

Validates the `Instance.args` tuple that `task.construct_requests()` produces. This tuple is the contract between task code and model code — its length, order, and types must be exact.

The tuple shape varies by task type and model type:

| Task Type | Model Type | `output_type` | Tuple Shape |
|-----------|------------|---------------|-------------|
| `ConfigurableTask` | Simple | `generate_until` | `(prompt, gen_kwargs, doc_to_visual, doc_id, task, split)` |
| `ConfigurableTask` | Simple | `loglikelihood` | `(context, target, doc_to_visual, doc_id, task, split)` |
| `ConfigurableMessagesTask` | Chat | `generate_until` | `(doc_to_messages, gen_kwargs, doc_id, task, split)` |
| Multi-round | Simple | `generate_until_multi_round` | `(context, gen_kwargs, doc_to_visual, doc_to_text_multi_round, doc_id, task, split)` |
| Agentic | Simple | `generate_until_agentic` | `(context, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split)` |

The tests use mock tasks with controlled YAML configs to verify each combination. They check tuple length, the types of each element (callable, dict, int, str), and specific values where applicable.

> Historical note: Many production bugs have originated from tuple order mismatches. A model unpacking `(prompt, gen_kwargs, doc_to_visual, ...)` receives garbage if the task emits `(doc_to_messages, gen_kwargs, doc_id, ...)` instead.

---

### Message Protocol Layer

Protocol tests verify that multimodal inputs are correctly structured and media can be extracted. For example, images and videos from a mixed message are separated into typed lists:

```python
def test_extract_media_mixed_content(sample_image):
    messages = _build_chat_messages([{
        "role": "user",
        "content": [
            {"type": "image", "url": sample_image},
            {"type": "video", "url": "/path/to/video.mp4"},
            {"type": "text", "text": "Describe both."},
        ],
    }])

    images, videos, audios = messages.extract_media()
    assert len(images) == 1 and isinstance(images[0], Image.Image)
    assert len(videos) == 1 and videos[0] == "/path/to/video.mp4"
    assert audios == []
```

#### `test/eval/test_protocol.py`

Tests `ChatMessages`, the unified message protocol that all chat models consume. Every multimodal input (image, video, audio, text) flows through this structure.

The test covers:

- **Construction** — Single-turn and multi-turn messages with mixed content types (text + image + video in one message).
- **`extract_media()`** — Correctly separates images, videos, and audios into three lists while preserving order. Handles edge cases: no media, multiple media per message, media-only messages with no text.
- **`to_hf_messages()`** — Converts to HuggingFace format where media URLs are replaced with `<image>`/`<video>`/`<audio>` placeholders and text segments are concatenated.
- **Edge cases** — Empty message list, empty content list, messages with only media and no text, unknown content types.

---

### Execution and Caching Layer

Execution tests verify the agentic multi-round loop and cache hit/miss lifecycle. For example, the loop terminates when `doc_to_text` signals completion:

```python
def test_agentic_single_round_terminal():
    model = _make_simple_model(["model_reply"])
    instance = _make_agentic_instance(
        doc_to_text=_terminal_doc_to_text,  # signals terminal after round 0
    )

    _run_generate_until_agentic([instance], model, max_agentic_steps=5)

    output = json.loads(instance.resps[0])
    assert output["rounds"] == 1
    assert output["last_model_output"] == "model_reply"
```

Cache tests verify that deterministic requests are stored and replayed:

```python
def test_cache_hit_miss_lifecycle(self):
    requests = [_gen_request(prompt="What is 2+2?", temperature=0)]

    # First run: cache miss, model is called
    cache = self._open_cache()
    to_run, cached = cache.filter_cached(requests)
    assert len(to_run) == 1 and len(cached) == 0
    cache.store(to_run, ["4"])
    cache.close()

    # Second run: cache hit, model is skipped
    cache = self._open_cache()
    to_run, cached = cache.filter_cached(requests)
    assert len(to_run) == 0 and len(cached) == 1
```

#### `test/eval/test_evaluator.py`

Tests `_run_generate_until_agentic`, the multi-round agent evaluation loop in `evaluator.py`. This function orchestrates:

1. Send prompt to model.
2. Call `doc_to_text` with the model's output to get the next prompt (or a terminal signal).
3. Repeat until terminal or `max_agentic_steps` is reached.
4. Serialize the full round history as JSON.

The test covers:

- **Normal termination** — `doc_to_text` returns `terminal=True` after one round. Output JSON contains `rounds`, `last_model_output`, and `round_info`.
- **Max steps enforcement** — When `max_agentic_steps=2`, the loop stops after two rounds even if `doc_to_text` never signals terminal.
- **Simple vs. chat model paths** — Both `is_simple=True` and `is_simple=False` models are exercised.
- **Cache integration** — With a `ResponseCache`, the second call with the same prompt skips the model entirely.

#### `test/cache/test_response_cache.py`

The largest test file (~540 lines). It covers the full lifecycle of `ResponseCache`, the SQLite-backed response cache that avoids re-running deterministic model calls.

**Pure function tests (no I/O):**

- `is_deterministic()` — `temperature=0` is deterministic; `temperature>0`, `do_sample=True`, or `n>1` are not. `loglikelihood` requests are always deterministic.
- `compute_cache_key()` — Different prompts, different idx, different tasks, different fingerprints all produce different keys. Float/int normalization (`0.0 == 0`).
- `extract_gen_kwargs()` — Extracts the gen_kwargs dict from an Instance's args tuple.
- `_is_valid_response()` — Rejects `None`, empty strings, whitespace, and malformed loglikelihood tuples.

**Integration tests (with temp SQLite DB):**

- **Hit/miss lifecycle** — First run: all misses, responses stored. Second run: all hits, model not called. Partial overlap: only missing requests sent to model.
- **Non-deterministic bypass** — Requests with `temperature > 0` or `do_sample=True` are never cached. They always go to the model.
- **Anti-poisoning** — Non-deterministic responses from run 1 do not leak into run 2.
- **Audit log** — Every request (cached or not) produces a JSONL audit entry with deterministic flag, cache key, fingerprints, and serialized response.
- **Crash recovery** — If the SQLite DB is deleted but the JSONL audit log survives, reopening the cache replays the log and recovers all entries.
- **Multi-rank isolation** — Each distributed rank writes to its own DB file. `merge_shards()` combines them, deduplicating overlapping keys.
- **Model fingerprint isolation** — Different models (different fingerprint strings) never share cache entries.
- **Stats accuracy** — Hit/miss/skip counters and `total_cached_entries` remain correct across close/reopen cycles.
- **Large batch** — 1,000 requests write and read-back within 10 seconds (performance guard).

#### `test/cache/test_agentic_response_cache.py`

Two focused tests for the intersection of agentic evaluation and caching:

- Same prompt + same doc → cache hit on second call (model called once).
- Same doc but different prompt → no collision (model called twice).

#### `test/cache/test_determinism_parametrized.py`

The canonical parametrized test suite for `is_deterministic()`. Contains 14 cases covering all combinations of request type, temperature, `do_sample`, and multi-return keys.

---

### Token Counting Layer

Token counting tests verify that diverse model return formats are normalized into a consistent `(text, TokenCounts)` pair:

```python
def test_unwrap_plain_string():
    text, tc = unwrap_generation_output("hello world")
    assert text == "hello world"
    assert tc is None

def test_unwrap_generation_result():
    gr = GenerationResult(
        text="answer",
        token_counts=TokenCounts(input_tokens=100, output_tokens=20),
    )
    text, tc = unwrap_generation_output(gr)
    assert text == "answer"
    assert tc.input_tokens == 100
    assert tc.output_tokens == 20
```

#### `test/eval/test_token_counts.py`

Tests the per-sample token counting infrastructure that feeds into usage tracking and efficiency metrics.

- **`TokenCounts`** — Dataclass with `input_tokens`, `output_tokens`, `reasoning_tokens`. All default to `None`. `to_dict()` omits `None` fields.
- **`GenerationResult`** — Wraps a text string with optional `TokenCounts`. Models that support usage reporting return this instead of a plain string.
- **`unwrap_generation_output()`** — Normalizes diverse return formats into `(text, TokenCounts | None)`. Handles: plain strings, `GenerationResult`, `(str, TokenCounts)` tuples, `(str, dict)` tuples, lists, and fallback `str()` for unknown types.
- **`Instance.token_counts`** alignment — The `token_counts` list stays in sync with the `resps` list as responses are appended.
- **`ResponseCache` integration** — `_extract_cacheable()` reduces `GenerationResult` to plain text for storage. `_is_valid_response()` accepts non-empty `GenerationResult` and rejects empty ones.

---

### Metrics Layer

Metrics tests verify that sample-level token counts aggregate correctly into task-level efficiency summaries:

```python
def test_efficiency_metrics_tokens_per_correct(base_results):
    summary = build_efficiency_summary(base_results)

    assert summary["overall"]["total_input_tokens"] == 150.0   # 100 + 50
    assert summary["overall"]["total_output_tokens"] == 50.0   # 20 + 30
    assert summary["overall"]["total_correct_score"] == 1.0    # only first sample scored 1
    assert summary["overall"]["tokens_per_correct_answer"] == 50.0  # 50 output / 1 correct
```

#### `test/eval/test_efficiency_metrics.py`

Tests `build_efficiency_summary()`, which aggregates sample-level token counts into task-level and overall efficiency metrics.

- **Normal case** — Two samples with known token counts and scores. Verifies `total_input_tokens`, `total_output_tokens`, `total_correct_score`, and `tokens_per_correct_answer`.
- **Empty samples** — Returns empty dict (no crash).
- **Score key fallback** — When the configured `score_key` is missing from a sample, falls back to the `acc` field.
- **Non-dict token entries** — `None` and string entries in `token_counts` are silently skipped.
- **Zero correct** — `tokens_per_correct_answer` is `None` (avoids division by zero).

#### `test/eval/test_usage_metrics.py`

End-to-end API integration test for usage tracking. **This test is auto-skipped** unless `OPENROUTER_API_KEY` is set in the environment. It is marked with `@pytest.mark.api` and `@pytest.mark.slow`.

When run, it:

1. Calls `simple_evaluate()` with an OpenRouter model on MME (limit=2).
2. Verifies the `usage` dict has the expected structure (`total`, `by_task`, `by_source`, `budget_exceeded`).
3. Checks that token counts are positive and `n_api_calls == 2`.
4. Checks that `total_tokens == input_tokens + output_tokens + reasoning_tokens`.
5. Tests budget enforcement: with `max_tokens=1`, the evaluation completes but `budget_exceeded` is `True`.

---

## Markers and Auto-Skip

Custom markers are defined in `pyproject.toml` and auto-skip logic lives in `test/conftest.py`:

| Marker | Condition to Run | Auto-Skip When |
|--------|-----------------|----------------|
| `@pytest.mark.gpu` | `torch.cuda.is_available()` | No GPU detected |
| `@pytest.mark.api` | `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY` set | No API key in env |
| `@pytest.mark.slow` | Always runs (informational) | — |

Tests without markers always run. The auto-skip ensures the default `pytest` invocation succeeds on any developer machine.

## Adding New Tests

### For a new task

1. Add the task name to `MAINSTREAM_TASKS` in `test/eval/test_task_pipeline.py` if it should be part of pipeline validation.
2. If the task is a classic benchmark, add a prompt stability snapshot:
   - Create a test case in `test/eval/prompt_stability/test_prompt_stability.py` with a synthetic doc.
   - Run `pytest test/eval/prompt_stability/ --update-snapshots` to generate the golden file.
   - Commit the new `.json` snapshot.

### For a new model

Model-specific end-to-end tests are not currently in the suite (removed during cleanup). When adding them back, follow these conventions:

- Place in `test/models/` (not `test/eval/`).
- Mark with `@pytest.mark.gpu` and `@pytest.mark.slow`.
- Test model loading and a single `generate_until` call with a tiny limit.
- Do not test task-specific logic — that belongs in task tests.

### For a new CLI argument

Add a test in `test/eval/test_cli_parse_args.py` following the existing pattern: construct `argv`, patch `sys.argv`, call `parse_eval_args()`, assert the value.

### For cache behavior changes

Add test cases to `test/cache/test_response_cache.py`. Use the `tmp_path` fixture for automatic temp directory setup/teardown.

## File Reference

```
test/
├── conftest.py                              # Shared fixtures, markers, --update-snapshots
├── __init__.py                              # Package marker
├── README.md                                # This file
│
├── cache/
│   ├── __init__.py
│   ├── test_response_cache.py               # ResponseCache full lifecycle (~36 tests)
│   ├── test_agentic_response_cache.py       # Agentic + cache integration (2 tests)
│   └── test_determinism_parametrized.py     # is_deterministic parametrized (14 tests)
│
├── cli/
│   ├── __init__.py
│   └── test_cli_dispatch_parametrized.py    # CLI dispatch router, parametrized (17 tests)
│
├── eval/
│   ├── __init__.py
│   ├── test_protocol.py                     # ChatMessages protocol (32 tests)
│   ├── test_construct_requests.py           # Instance.args tuple shapes (23 tests)
│   ├── test_evaluator.py                    # Agentic evaluation loop (15 tests)
│   ├── test_task_pipeline.py                # Task registration + YAML + utils (~20 tests)
│   ├── test_benchmark_aliases.py            # Alias group resolution (2 tests)
│   ├── test_new_benchmarks_tasks.py         # New benchmark validation (~15 tests)
│   ├── test_cli_parse_args.py               # CLI argument parsing (2 tests)
│   ├── test_token_counts.py                 # Token counting infrastructure (~20 tests)
│   ├── test_efficiency_metrics.py           # Efficiency metric aggregation (5 tests)
│   ├── test_usage_metrics.py                # API usage tracking (8 tests, auto-skip)
│   │
│   └── prompt_stability/
│       ├── __init__.py
│       ├── test_prompt_stability.py         # Snapshot comparison (22 tests)
│       └── snapshots/                       # Golden prompt files (11 JSON files)
│           ├── mmmu_val__mc.json
│           ├── mmmu_val__open.json
│           ├── mmbench_en_dev__hint.json
│           ├── mmbench_en_dev__nohint.json
│           ├── hallusion_bench_image.json
│           ├── ocrbench.json
│           ├── mmlongbench_doc.json
│           ├── vsibench__mca.json
│           ├── vsibench__na.json
│           ├── mmvp.json
│           └── blink__art_style.json
│
└── models/
    ├── __init__.py
    └── test_model_registry_v2.py            # Model registry resolution (~12 tests)
```

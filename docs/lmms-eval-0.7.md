# LMMs-Eval v0.7

## Overview

v0.7 focuses on making lmms-eval easier to use, share, and reproduce. The theme is **operational simplicity** - fewer flags to remember, fewer things that can go wrong, and a single file that captures your entire experiment.

---

## 1. Better One-Line Evaluation

Running an evaluation used to mean assembling a long command with many flags. Sharing that command with a teammate meant copy-pasting a fragile shell one-liner, hoping the environment variables were set, and trusting the args were right. Reproducing a result from a paper meant reverse-engineering the setup from a results JSON that only stored a few fields.

v0.7 replaces all of that with a single YAML file:

```bash
python -m lmms_eval --config configs/my_experiment.yaml
```

One file. Everything is in it - model, tasks, generation parameters, environment variables. No separate `export` commands, no long CLI flags. Ship the YAML, reproduce the result.

### 1.1 What Goes in the YAML

A config file maps directly to CLI arguments, plus an optional `env` section for environment variables:

```yaml
# configs/example_local.yaml
env:
  HF_HOME: "${HF_HOME:-~/.cache/huggingface}"
  HF_HUB_ENABLE_HF_TRANSFER: "1"

model: qwen2_5_vl
model_args: "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
tasks: "mme,mmmu_val"
batch_size: 1
num_fewshot: 0
seed: "42,42,42,42"
gen_kwargs: "temperature=0,max_new_tokens=1024"
output_path: "results/"
log_samples: true
```

For API models with credentials:

```yaml
# configs/example_api.yaml
env:
  OPENAI_API_KEY: "${OPENAI_API_KEY}"
  HF_HOME: "${HF_HOME:-~/.cache/huggingface}"

model: openai
model_args: "model=gpt-4o,max_retries=5"
tasks: "mme,mmmu_val"
batch_size: 1
gen_kwargs: "temperature=0,max_new_tokens=1024"
output_path: "results/"
log_samples: true
```

For batch evaluation (multiple models in one run):

```yaml
# configs/example_batch.yaml
- model: qwen2_5_vl
  model_args: "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
  tasks: "mme"
  batch_size: 1
  output_path: "results/qwen25vl/"
  log_samples: true

- model: qwen3_vl
  model_args: "pretrained=Qwen/Qwen3-VL-4B-Instruct,device_map=auto"
  tasks: "mme"
  batch_size: 1
  output_path: "results/qwen3vl/"
  log_samples: true
```

### 1.2 Environment Variables

The `env` section sets environment variables before evaluation starts. This solves the "works on my machine" problem - credentials and paths are part of the config, not floating in someone's `.bashrc`.

```yaml
env:
  OPENAI_API_KEY: "sk-..."           # Literal value
  HF_TOKEN: "${HF_TOKEN}"            # Expand from shell environment
  HF_HOME: "${HF_HOME:-/data/cache}" # Expand with default fallback
```

**Variable expansion**: Values containing `${VAR}` are expanded using the current shell environment. The `${VAR:-default}` syntax provides fallback defaults. This lets you keep secrets out of the YAML while still documenting which variables are needed.

**Sensitive value masking**: Keys containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` are masked in log output (`Config env: OPENAI_API_KEY=********`). The actual values are set correctly in `os.environ`; only the log is masked.

### 1.3 CLI Override Priority

The priority chain is: **defaults < YAML < CLI**. CLI arguments always win.

```bash
# YAML says limit: null, but you want a quick test
python -m lmms_eval --config configs/example_local.yaml --limit 5

# YAML says batch_size: 1, override for faster throughput
python -m lmms_eval --config configs/example_local.yaml --batch_size 4
```

This lets you keep a "canonical" config in the YAML and override individual values at the command line without editing the file. The override detection works by comparing each CLI argument against argparse defaults - only arguments that differ from defaults are treated as explicit overrides.

### 1.4 Schema Validation

Unknown keys in the YAML now raise an error with the list of valid keys:

```
ValueError: Unknown keys in config file: ['modle', 'taks'].
Valid keys are: ['batch_size', 'config', 'device', 'gen_kwargs', 'limit', 'log_samples', 'model', 'model_args', ...]
```

Previously, typos like `modle` instead of `model` were silently accepted via `setattr` and ignored at runtime, leading to confusing evaluation failures.

### 1.5 Full Experiment Reproducibility

Results JSON now includes `resolved_cli_args` - the complete resolved configuration after merging defaults, YAML, and CLI overrides:

```json
{
  "config": {
    "model": "qwen2_5_vl",
    "model_args": "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto",
    "resolved_cli_args": {
      "model": "qwen2_5_vl",
      "model_args": "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto",
      "tasks": "mme,mmmu_val",
      "batch_size": 1,
      "num_fewshot": 0,
      "seed": "42,42,42,42",
      "gen_kwargs": "temperature=0,max_new_tokens=1024",
      "output_path": "results/",
      "log_samples": true,
      "limit": null,
      "..."
    }
  }
}
```

This means you can reconstruct the exact YAML config from any results file. No more guessing what flags were used.

### 1.6 Web UI Integration

The Web UI now supports YAML import and export:

- **Export**: Configure an evaluation in the Web UI, then click "Export YAML" to download the config as a `.yaml` file. Share it with teammates or commit it to your repo.
- **Import**: Click "Import YAML" to load a config file into the Web UI form. Useful for reviewing or tweaking a config before running.

The `env` section maps to the Web UI's environment variables field. The round-trip is lossless - export a YAML, import it back, and the form state is identical.

### 1.7 Example Configs

Three example configs are included in `configs/`:

| File | Use Case |
|------|----------|
| `configs/example_local.yaml` | Local GPU evaluation (Qwen2.5-VL) |
| `configs/example_api.yaml` | API model evaluation (OpenAI-compatible) |
| `configs/example_batch.yaml` | Multiple models in a single run |

Use them as templates:

```bash
cp configs/example_local.yaml configs/my_experiment.yaml
# Edit to your needs, then run
python -m lmms_eval --config configs/my_experiment.yaml
```

---

## 2. Pipeline-Level Reasoning Tag Stripping

Reasoning models (Qwen3-VL, DeepSeek-R1, QwQ, etc.) emit `<think>...</think>` blocks as part of their generated text. Without stripping, these tokens pollute scoring on standard benchmarks. Previously, only 6 model files had ad-hoc handling; vLLM, SGLang, and OpenAI backends had zero protection.

v0.7 moves reasoning tag stripping into the evaluator pipeline itself, so it works uniformly across all model backends.

### 2.1 How It Works

Stripping happens in `evaluator.py` **after** the filter pipeline and **before** `process_results()`. Both the raw and cleaned outputs are preserved:

```
Model.generate_until()  ->  raw output (with <think>)
    |
Filter pipeline  ->  filtered_resps (initial)
    |
strip_reasoning_tags()  (in evaluator.py)
    |-> resps = pre-strip value (preserved for analysis)
    |-> filtered_resps = clean text (used for scoring)
    |
process_results(doc, filtered_resps)  ->  metric scores
```

This means:
- **Models return raw output** - no model file needs to handle tag stripping anymore.
- **Scoring is clean** - `process_results()` never sees `<think>` tokens.
- **Analysis is preserved** - the raw chain-of-thought is still available in `--log_samples` output.

### 2.2 Usage

Stripping is enabled by default with `<think>...</think>` tags:

```bash
# Default behavior - stripping enabled
python -m lmms_eval --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
    --tasks mme --limit 8 --log_samples

# Disable stripping
python -m lmms_eval --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
    --tasks mme --reasoning_tags none

# Custom tag pairs (JSON format)
python -m lmms_eval --model qwen3_vl --tasks mme \
    --reasoning_tags '[["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]'
```

### 2.3 Per-Task Override

Tasks can override the CLI setting via the `reasoning_tags` field in their YAML config. Task-level config takes priority over the CLI flag.

```yaml
# In your task YAML
reasoning_tags: [["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]
```

Set to `none` or `false` to disable for a specific task.

### 2.4 JSONL Log Output Fields (`--log_samples`)

When `--log_samples` is passed, each JSONL line contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `int` | Index of the document within the dataset split. |
| `input` | `str` | The prompt / context string fed to the model. |
| `target` | `str` | Ground-truth answer from `doc_to_target()`. |
| `resps` | `list` | Raw model output **before** reasoning tag stripping. Preserves `<think>` blocks intact for chain-of-thought analysis. Omitted when identical to `filtered_resps`. |
| `filtered_resps` | `list` | Model output **after** filter pipeline + reasoning tag stripping. This is what was actually scored by `process_results()`. |
| `doc_hash` | `str` | SHA-256 hash of the document JSON for deduplication and cross-run alignment. |
| `<metric>` | `float/int` | Per-sample metric scores from `process_results()` (e.g., `exact_match`, `acc`). Keys depend on the task. |

**Example JSONL record** (reasoning model with `--log_samples`):

```json
{
  "doc_id": 0,
  "input": "What is shown in this image?\nAnswer with a single word.",
  "target": "cat",
  "resps": [["<think>\nThe image shows a small furry animal sitting on a windowsill...\n</think>\ncat"]],
  "filtered_resps": ["cat"],
  "doc_hash": "a1b2c3d4e5...",
  "exact_match": 1.0
}
```

- **`resps`** - useful for debugging and analyzing model reasoning behavior. Contains the full chain-of-thought.
- **`filtered_resps`** - the canonical scored output. Use this when computing or verifying metrics.
- When no stripping occurred (non-reasoning model or stripping disabled), `resps` is omitted if identical to `filtered_resps` to save space.

### 2.5 Implementation Details

| File | Change |
|------|--------|
| `lmms_eval/api/reasoning.py` (NEW) | `strip_reasoning_tags()` and `parse_reasoning_tags_config()` |
| `lmms_eval/evaluator.py` | Strip logic before scoring, dual storage in JSONL |
| `lmms_eval/__main__.py` | `--reasoning_tags` CLI argument |
| `lmms_eval/api/task.py` | Per-task `reasoning_tags` field in `TaskConfig` |
| `lmms_eval/api/instance.py` | `raw_filtered_resps` dict for pre-strip preservation |
| 6 model files | Removed ad-hoc `parse_reasoning_model_answer` calls |

---

## 3. Async OpenAI: Refactored Concurrency Control & `message_format`

The `async_openai` model backend received two changes: internal refactoring for maintainability, and a `message_format` parameter that replaces the previous `is_qwen3_vl` flag and the separate `async_openai_qwen3_vl` model class. (#1102)

### 3.1 `message_format` Parameter

Different model families served behind OpenAI-compatible APIs may require different message serialization. For example, Qwen3-VL needs per-frame timestamps prepended to video frames, while the standard OpenAI format sends frames as plain base64 images.

Previously this was handled by an `is_qwen3_vl` boolean flag, then by a separate model class (`async_openai_qwen3_vl`). Both approaches scale poorly - every new format would require a new flag or a new file + class + registry entry.

v0.7 replaces this with a single `message_format` parameter on `async_openai`:

```bash
# Standard OpenAI format (default)
python -m lmms_eval --model async_openai \
    --model_args pretrained=gpt-4o,message_format=openai \
    --tasks mme

# Qwen3-VL format (adds per-frame timestamps for video)
python -m lmms_eval --model async_openai \
    --model_args pretrained=Qwen/Qwen3-VL-72B,message_format=qwen3_vl \
    --tasks video_mme
```

Adding a new format requires only an `elif` in `prepare_messages()` and a corresponding `to_*_messages()` method in `ChatMessages` - no new files or registry changes.

### 3.2 Refactored Concurrency Control

The `generate_until()` method was a single 130-line function containing retry logic, adaptive concurrency control, and request scheduling all interleaved. It has been decomposed into focused methods:

| Method | Responsibility |
|--------|----------------|
| `_build_video_kwargs()` | Construct video processing parameters from model config |
| `prepare_messages()` | Dispatch to format-specific message serialization |
| `_get_initial_concurrency()` | Compute starting concurrency from CPU count and adaptive config |
| `_compute_dispatch_order()` | Sort requests by prefix hash for cache locality |
| `_process_with_retry()` | Execute a single request with retry and backoff |
| `_update_concurrency()` | Adjust concurrency based on failure/latency signals |
| `_run_scheduling_loop()` | Main async scheduling loop with slot refill |

The `generate_until()` method is now 8 lines:

```python
async def run():
    pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
    current_concurrency = self._get_initial_concurrency()
    dispatch_order = self._compute_dispatch_order(requests)
    res = await self._run_scheduling_loop(requests, dispatch_order, pbar, current_concurrency)
    pbar.close()
    return res
```

Concurrency tracking state is encapsulated in `_AdaptiveConcurrencyTracker` (a dataclass) instead of scattered `nonlocal` variables in nested closures.

---

## 4. Flattened JSONL Log Output

When `--log_samples` is enabled, the per-sample JSONL files previously wrote `resps` and `filtered_resps` as doubly-nested lists:

```json
{"resps": [["The answer is cat"]], "filtered_resps": [["cat"]]}
```

The outer list exists because the evaluator groups multiple `Instance` objects per document (e.g., one per choice in multiple-choice tasks). For the dominant `generate_until` output type, there is always exactly one Instance per document, making the outer list redundant.

v0.7 flattens the outer list at serialization time when it contains only a single element:

```json
{"resps": ["The answer is cat"], "filtered_resps": ["cat"]}
```

### 4.1 When Flattening Applies

| Output Type | Instances per Doc | Before | After |
|-------------|-------------------|--------|-------|
| `generate_until` | 1 | `[["text"]]` | `["text"]` |
| `generate_until_multi_round` | 1 | `[["text"]]` | `["text"]` |
| `loglikelihood` (MCQ) | N (one per choice) | `[["a"], ["b"], ...]` | `[["a"], ["b"], ...]` (unchanged) |

Flattening only removes the outer wrapper when there is exactly one Instance. Multi-choice tasks with multiple Instances per document are left untouched.

### 4.2 Deduplication with Flattened Format

The existing dedup logic (omit `resps` when identical to `filtered_resps`) continues to work with the flattened format. After flattening, the two fields are compared directly - if they match, `resps` is omitted from the JSONL record to save space.

### 4.3 Implementation

The flatten happens in `evaluation_tracker.py` during JSONL serialization, not in the evaluator core. In-memory data structures (`logged_samples`) retain the original nested format so that existing consumers (wandb logger, logging utilities) continue to work without changes.
---

## 5. Safety and Red-Teaming Baseline (JailbreakBench)

v0.7 adds a first safety/red-teaming task group based on JailbreakBench behaviors. This addresses a long-standing gap where lmms-eval had no built-in safety benchmark for jailbreak robustness and over-refusal analysis.

### 5.1 New Task Group

- `safety_redteam` (group)
  - `safety_jailbreakbench_harmful`
  - `safety_jailbreakbench_benign`

Dataset source: `JailbreakBench/JBB-Behaviors` (`behaviors` config, harmful + benign splits).

### 5.2 Reported Metrics

**Harmful split**:

- `jailbreak_asr` - attack success proxy (lower is better)
- `refusal_rate` - refusal behavior under harmful prompts (higher is better)
- `toxicity_score` - toxicity estimate in [0, 1] (lower is better)
- `content_filter_rejection_rate` - policy/filter rejection frequency (higher is better)
- `demographic_refusal_rate`
- `non_demographic_refusal_rate`

**Benign split**:

- `over_refusal_rate` - benign prompts incorrectly refused (lower is better)
- `benign_toxicity_score` - toxicity estimate on benign prompts (lower is better)
- `content_filter_rejection_rate`
- `demographic_refusal_rate`
- `non_demographic_refusal_rate`

### 5.3 Toxicity Backends

Toxicity scoring supports two modes:

1. Perspective API when `PERSPECTIVE_API_KEY` is configured
2. Offline keyword heuristic fallback when API is unavailable

This keeps safety evaluation usable in both cloud and offline environments.

### 5.4 Usage

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks safety_redteam \
  --batch_size 1 \
  --limit 20
```

---

## 6. Skill-Based Agent Workflows (New)

v0.7 also standardizes how coding agents can learn and orchestrate lmms-eval workflows through the repository skill:

- `skills/lmms-eval-guide/SKILL.md`
- `skills/lmms-eval-guide/references/models.md`
- `skills/lmms-eval-guide/references/tasks.md`
- `skills/lmms-eval-guide/references/api-server.md`

This turns lmms-eval from "a set of docs" into "a reusable operational skill" for agents: discover the right integration path, apply the correct file-level patterns, and schedule evaluation jobs safely.

### 6.1 Add New Models and Tasks via Skill References

For extension work, the skill references already define the recommended implementation paths:

- **New model integration** -> `references/models.md`
  - Chat-first model template (`is_simple = False`)
  - Request unpacking contract (`request.args` shape)
  - Registration in `lmms_eval/models/__init__.py`
  - Minimal verification commands (`--limit 5` / `--verbosity DEBUG`)

- **New task/benchmark integration** -> `references/tasks.md`
  - YAML-first task definition (auto-registered from `tasks/`)
  - `doc_to_messages` + fallback `doc_to_visual` / `doc_to_text`
  - `process_results` + `metric_list` contract
  - Advanced patterns (`include`, `group`, `cluster_key`, LLM-as-judge)

Agent-level behavior should be:

1. Resolve whether change is model-side, task-side, or both.
2. Load the matching skill reference.
3. Follow existing code patterns in nearby chat/simple models and task YAMLs.
4. Run a small-sample verification command before broader evaluation.

This keeps model/task additions consistent with lmms-eval internals while reducing trial-and-error.

### 6.2 Insert lmms-eval into Training Jobs via HTTP Service

For training-time evaluation orchestration, use the eval server workflow from `references/api-server.md`.

Core pattern:

1. Start HTTP eval server (`launch_server(ServerArgs(...))`) on dedicated eval resources.
2. During training, submit non-blocking jobs with `EvalClient.evaluate(...)`.
3. Continue training immediately; poll or wait by `job_id` later.
4. Collect metrics asynchronously for checkpoint selection, regression alerts, and model ranking.

Key endpoints for job orchestration:

- `POST /evaluate` - submit evaluation jobs
- `GET /jobs/{job_id}` - query status/results
- `GET /queue` - inspect scheduler backlog
- `GET /tasks` and `GET /models` - runtime capability discovery

This service mode is the recommended way to decouple training and evaluation in v0.7-era workflows.

### 6.3 HTTP Service as an Operational Primitive

Treat the eval server as infrastructure, not only a convenience API:

- **Queue-safe GPU usage** via scheduler-managed jobs
- **Asynchronous checkpoint evaluation** without blocking trainer processes
- **Multi-team reproducibility** through stable request payloads and job IDs
- **Operational visibility** through `/queue` and job lifecycle states

Security reminder remains unchanged: run in trusted environments, and add authentication/rate limiting/network isolation before exposing beyond internal boundaries.

### 6.4 Suggested Agent Dispatch Strategy

When agents orchestrate lmms-eval tasks, a practical routing strategy is:

- **Task/model extension** -> load `lmms-eval-guide` + `references/models.md` / `references/tasks.md`
- **Training integration** -> load `references/api-server.md` first
- **Quick validation** -> run `--limit` smoke tests before full benchmark submission
- **Scalable evaluation** -> use HTTP jobs for long-running or periodic evaluation loops

This gives a clear split between development-time edits (model/task code) and runtime-time scheduling (HTTP jobs).

---

## 7. Image/Video I/O Throughput Upgrade with Long-Video Reproducibility

This update consolidates image encoding in shared helpers and optimizes video decode hot paths while preserving task-facing semantics.

### 7.1 What Was Implemented

- shared image encoding helper integration across protocol and simple adapters
- path-metadata keyed image encode cache for repeated path inputs
- video decode path upgrades in `lmms_eval/models/model_utils/load_video.py`
  - PyAV `thread_type="AUTO"`
  - stream fallback `seek(0)` before packet decode
  - set-membership lookup in stream frame selection
  - preallocated output array fill (replace list + `np.stack` path)
  - configurable decord threads via `LMMS_VIDEO_DECORD_THREADS`

### 7.2 Semantic-Parity Guard for `load_video` Resize

To avoid input-semantics drift, `resize_strategy="resize"` behavior is kept aligned with prior behavior (direct target-size resize) and validated against `dev-v0d7`.

Parity evidence artifact:

- `/tmp/load_video_parity_compare.json`

Observed parity checks:

- PNG first-frame size match: `True`
- PIL first-frame size match: `True`
- PNG first-frame hash match: `True`

### 7.3 Real-Model LongVideoBench Replay (Long Samples)

Provider-backed replay was run on `longvideobench_val_v` using OpenRouter (`bytedance-seed/seed-1.6-flash`) with long samples (`limit=8`, `max_frames_num=4`, `max_image_size=512`).

Artifacts:

- baseline: `/tmp/video-real-model/mainbench_real_r3.json`
- optimized: `/tmp/video-real-model/feat_real_postfix_r1.json`
- optimized rerun: `/tmp/video-real-model/feat_real_postfix_r2.json`
- summary: `/tmp/video-real-model/summary_real_postfix_compare.json`

Results:

- aggregate score reproducibility: `0.5 -> 0.5 -> 0.5` (baseline, optimized run1, optimized run2)
- decode latency: `2.7902s -> 1.0227s` (`-63.35%`, speedup `+172.82%`)
- optimized A/A decode variation: `-3.05%`

Prediction-level note:

- baseline vs optimized run1 per-sample parsed prediction match: `8/8`
- optimized run1 vs run2 per-sample match: `6/8`
- aggregate score remains unchanged; per-item drift is consistent with remote-provider nondeterminism

---

## 8. Lance-Backed Video Mode for MINERVA

v0.7 adds an optional Lance-backed video path for the MINERVA task so metadata and videos can be distributed through Hugging Face in a single reproducible package.

### 8.1 Dataset Package on Hugging Face

MINERVA is published as:

- `lmms-lab-eval/minerva`

with two key assets:

- `minerva.json` - task metadata used by `minerva.yaml`
- `data/train.lance` - Lance table with one row per `video_id`

The Lance table stores video bytes in `video_blob` with blob encoding metadata (`lance-encoding:blob=true`) so samples can be fetched by row ID through `take_blobs`.

### 8.2 Build Lance Table from Local Downloads

Use the conversion script:

```bash
uv run --with pylance --with pyarrow python tools/minerva_to_lance.py \
  --metadata-json data/minerva/minerva.json \
  --videos-dir data/minerva/videos \
  --output data/minerva_hf_package/data/train.lance \
  --batch-size 6
```

This produces a Lance schema with:

- `video_id`
- `youtube_url`
- `video_ext`
- `video_size_bytes`
- `video_blob`

Dependency note: install `pylance` (module import name: `lance`) plus `pyarrow` for Lance-mode usage.

### 8.3 Runtime Resolution Order in `lmms_eval/tasks/minerva/utils.py`

At evaluation time, MINERVA video resolution now uses this priority:

1. Local file lookup via `MINERVA_VIDEO_DIR`
2. Lance blob lookup via `MINERVA_LANCE_VIDEO_URI`
3. YouTube URL fallback (`https://www.youtube.com/watch?v=<video_id>`)

Lance mode variables:

```bash
export MINERVA_LANCE_VIDEO_URI="hf://datasets/lmms-lab-eval/minerva/data/train.lance"
export MINERVA_LANCE_VIDEO_ID_COLUMN="video_id"
export MINERVA_LANCE_VIDEO_BLOB_COLUMN="video_blob"
export MINERVA_LANCE_CACHE_DIR="~/.cache/lmms_eval/minerva_lance_videos"
export LANCE_IO_THREADS="64"
export LANCE_CPU_THREADS="8"
```

Optional local-first mode:

```bash
export MINERVA_VIDEO_DIR="/absolute/path/to/minerva/videos"
```

### 8.4 Run Example

```bash
uv run --with pylance --with pyarrow python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks minerva \
  --batch_size 1 \
  --limit 8
```

This setup keeps MINERVA reproducible in two modes: fully local video files or remote Lance blobs from the Hub.

### 8.5 Measuring Absolute Video-Resolution Latency Gains

To make Lance-mode improvements explicit, v0.7 includes a direct resolver benchmark:

```bash
uv run python tools/bench_minerva_video_resolution.py \
  --metadata-json data/minerva/minerva.json \
  --mode lance \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 200 \
  --sample-unique-video
```

For local-file baseline:

```bash
uv run python tools/bench_minerva_video_resolution.py \
  --metadata-json data/minerva/minerva.json \
  --mode local \
  --local-video-dir /absolute/path/to/minerva/videos \
  --limit 200
```

The benchmark reports absolute latency distribution for `minerva_doc_to_visual`:

- `startup_ms` (Lance resolver init cost)
- `cold_*` metrics (first pass, cache-miss heavy)
- `warm_*` metrics (second pass, cache-hit heavy)

For strict storage-path comparison inside the same model pipeline (no decode backend switching), use:

```bash
uv run python tools/bench_minerva_pipeline_latency.py \
  --local-video-dir /absolute/path/to/minerva/videos \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 100 \
  --batch-size 1 \
  --decode-num-frames 8
```

Practical interpretation:

- This comparison isolates storage mode while keeping decode unchanged in the model pipeline.
- On local pre-downloaded videos, local raw and Lance modes are often near-parity because decode cost dominates.
- Lance advantages become clearer in remote/object-storage access, cross-machine reproducibility, and repeated subset evaluation workflows.

Recommended reporting format for PRs and release notes:

- environment (CPU/GPU, storage, network)
- local mode vs Lance mode (both `cold_*` and `warm_*`)
- sample count and random seed

Implementation note: MINERVA Lance resolver now avoids eager full-table `video_id` scan at initialization and resolves rows via filtered scan on demand, which reduces startup overhead and better reflects real-world throughput in short eval runs.

For large-blob dataset build, use blob-oriented write settings in `tools/minerva_to_lance.py` (smaller rows-per-file and stable storage version) to improve practical scan/read behavior during evaluation.

# LMMs-Eval v0.7

v0.7 makes lmms-eval easier to operate and ready for production-tooling usage. The theme is **operational simplicity** for users and **pipeline maturity**, here's what we do.

## Table of Contents

- [Upgrading from v0.6](#upgrading-from-v06)
- [1. New Benchmark Tasks](#1-new-benchmark-tasks)
- [2. New Models](#2-new-models)
- [3. Skill-Based Agent Workflows](#3-skill-based-agent-workflows)
- [4. Agentic Task Evaluation](#4-agentic-task-evaluation)
- [5. Image/Video I/O Throughput Upgrade](#5-imagevideo-io-throughput-upgrade)
- [6. Lance-Backed Video Mode](#6-lance-backed-video-mode)
- [7. Safety and Red-Teaming Baseline](#7-safety-and-red-teaming-baseline)
- [8. Better One-Line Evaluation Support](#8-better-one-line-evaluation-support)
- [9. Efficiency Metrics Coverage](#9-efficiency-metrics-coverage)
- [10. Pipeline-Level Reasoning Tag Stripping](#10-pipeline-level-reasoning-tag-stripping)
- [11. Support customized message_format in async_openai](#11-support-customized-message_format-in-async_openai)
- [12. Flattened JSONL Log Output](#12-flattened-jsonl-log-output)
- [13. Bug Fixes](#13-bug-fixes)

---

## Upgrading from v0.6

> **Important:** Most workflows are backward-compatible. Two model-backend changes require attention before upgrading.

### Breaking Changes

- **`async_openai_qwen3_vl` model class removed.** Use `async_openai` with `message_format=qwen3_vl` instead:

  ```bash [Terminal]
  # Before (v0.6)
  python -m lmms_eval --model async_openai_qwen3_vl \
      --model_args pretrained=Qwen/Qwen3-VL-72B

  # After (v0.7)
  python -m lmms_eval --model async_openai \
      --model_args pretrained=Qwen/Qwen3-VL-72B,message_format=qwen3_vl
  ```

- **`is_qwen3_vl` flag removed** from `async_openai` model_args. Use `message_format=qwen3_vl` instead.

- **`parse_reasoning_model_answer` removed** from 6 model files. Reasoning tag stripping is now handled at the pipeline level (see [Section 10](#10-pipeline-level-reasoning-tag-stripping)). Remove any direct calls to this function.

### Backward-Compatible Renames

- `read_video_pyav` renamed to `read_video`. A backward-compatible alias is in place — existing code continues to work.

### Deprecations

No additional deprecations in v0.7. The v0.6 deprecation of `doc_to_visual` + `doc_to_text` for API models remains in effect. Use `doc_to_messages` + `ChatMessages` for new integrations.

### CLI Changes

v0.7 introduces a subcommand architecture (`eval`, `tasks`, `models`, `ui`, `serve`, `power`, `version`). The existing `python -m lmms_eval` invocation continues to work as before — subcommands are additive. See [docs/external_usage.md](../advanced/external_usage.md) for details.

---

## 1. New Benchmark Tasks

v0.7 adds 25+ benchmark tasks across eight domains:

| Domain | Tasks |
|--------|-------|
| **Document understanding** | OmniDocBench, MMLongBench, MMLongBench-Doc, DUDE, OfficeQA |
| **Video** | Neptune long-video benchmarks, TVBench, ViVerBench, EgoTempo |
| **Math & reasoning** | MathCanvas, MathKangaroo, VisuLogic, LLaVA-OV 1.5 RL reasoning collection |
| **Spatial & counting** | Point-Bench, CountBench, FSC-147 |
| **Knowledge & QA** | SimpleVQA, WorldVQA, MTVQA, HiPhO, MME-CC, VPCT, ZeroBench |
| **Agentic** | ARC-AGI-1, ARC-AGI-2, BrowseComp, Vending-Bench 2, τ2-Bench Telecom |
| **Audio** | AMI, CN College Listen MCQ, DREAM TTS MCQ, EuroPal ASR, Song Describer |
| **Safety** | JailbreakBench harmful + benign splits |

All tasks are auto-discovered from their YAML configs in `lmms_eval/tasks/`. No manual registration required. Run `python -m lmms_eval --tasks list` to see all available tasks.

---

## 2. New Models

| Model | Description |
|-------|-------------|
| **NanoVLM** | SigLIP2 + MLP projector + Qwen3-0.6B. Chat-style evaluation with async multi-GPU inference via job-queue dispatch. |
| **Async HF model** | Generic async multi-GPU worker backend for HuggingFace model families. Loads replicas on N GPUs with independent worker threads. |

---

## 3. Skill-Based Agent Workflows

Coding agents working on lmms-eval need to know where model files live, how task YAMLs are structured, what the registry contract looks like, and how to verify changes - context that is scattered across source files and docs. The skill system packages this into structured references so agents can load the right context at the right time instead of guessing.

v0.7 ships the skill as a set of files in the repository:

- `skills/lmms-eval-guide/SKILL.md` — entry point and routing logic
- `skills/lmms-eval-guide/references/models.md` — model integration patterns
- `skills/lmms-eval-guide/references/tasks.md` — task/benchmark definition patterns
- `skills/lmms-eval-guide/references/api-server.md` — HTTP eval server workflows
- `skills/lmms-eval-guide/references/workflows.md` — config, debugging, operational patterns

### 3.1 What the Skill References Cover

Each reference encodes the recommended implementation path for one concern:

- **Model integration** (`references/models.md`) — chat-first model template (`is_simple = False`), request unpacking contract (`request.args` shape), `message_format` parameter for API serialization dispatch, registry entry in `__init__.py`, v0.6 -> v0.7 migration, and verification commands (`--limit 5` / `--verbosity DEBUG`).

- **Task integration** (`references/tasks.md`) — YAML-first task definition (auto-registered from `tasks/`), `doc_to_messages` + fallback `doc_to_visual` / `doc_to_text`, `process_results` + `metric_list` contract, `reasoning_tags` per-task override, and advanced patterns (`include`, `group`, `cluster_key`, LLM-as-judge).

- **Training-time evaluation** (`references/api-server.md`) — start the HTTP eval server on dedicated resources, submit non-blocking jobs with `EvalClient.evaluate(...)` during training, poll or wait by `job_id`, collect metrics for checkpoint selection. The server provides queue-safe GPU usage, async checkpoint evaluation, and operational visibility via `/queue` and job lifecycle states.

- **Operational workflows** (`references/workflows.md`) — `--config` YAML usage, debugging steps, config structure and CLI override priority.

::tip
The intended workflow: resolve whether the change is model-side, task-side, or both; load the matching reference; follow existing code patterns in nearby files; run a small-sample verification before broader evaluation.
::

### 3.2 Agent Dispatch Strategy

When agents orchestrate lmms-eval tasks, use this routing to load the right reference:

| Scenario | Action |
|----------|--------|
| Task/model extension | Load `references/models.md` or `references/tasks.md` |
| YAML config setup | Load `references/workflows.md` for `--config` usage and config structure |
| Training integration | Load `references/api-server.md` first |
| Quick validation | Run `--limit` smoke tests before full benchmark submission |
| Scalable evaluation | Use HTTP jobs for long-running or periodic evaluation loops |
| Debugging failures | Load `references/workflows.md` for step-by-step debug workflow |

This separates development-time edits (model/task code) from runtime scheduling (HTTP jobs) and operational workflows (config, debugging).

---

## 4. Agentic Task Evaluation

v0.7 introduces `generate_until_agentic`, a new output type for evaluating models as tool-calling agents. The evaluator runs an iterative loop: on each step the model emits a `<tool_call>` or `<submit>` tag, the task's `doc_to_text` callback executes the tool against a deterministic simulator, and the evaluator feeds the result back as the next prompt. The loop terminates when the model submits a final answer or reaches `max_agentic_steps`.

### 4.1 How It Works

The pipeline lives in `_run_generate_until_agentic()` in `evaluator.py`. Each step:

1. The model generates a response (single `generate_until` call).
2. The task's `doc_to_text` callback parses the response for `<tool_call>` or `<submit>` tags.
3. If `<tool_call>`: the callback executes the tool against a Python simulator, updates internal state, and returns the next prompt with the tool result. The evaluator loops.
4. If `<submit>`: the callback returns a terminal signal with the final payload (success/failure, trace, state). The evaluator stops.
5. If neither: the callback returns an error hint and the loop continues.
6. If `max_agentic_steps` is reached without a submit, the evaluator emits a fallback payload with `"error": "max_agentic_steps_reached"` and the accumulated trace.

The `doc_to_text` function signature for agentic tasks:

```python
def doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None,
                round_idx=None, previous_round_info=None):
    # Returns either:
    # - str (next prompt)
    # - 5-tuple: (visuals, next_context, terminal_signal, updated_outputs, round_info)
```

### 4.2 Agentic Tasks

Two tasks validate the infrastructure with deterministic Python simulators (no external sandbox required):

| Task | Domain | Tools | Goal | Steps |
|------|--------|-------|------|-------|
| `vending_bench2` | Vending machine operation | `get_vending_status`, `set_price`, `restock`, `simulate_days` | Reach target cash and elapsed days | 10 |
| `tau2_bench_telecom` | Telecom customer support | `get_line_status`, `disable_airplane_mode`, `enable_roaming`, `reset_network` | Reach target device state | 8 |

Quick smoke test:

```bash
export OPENAI_API_KEY="your-openrouter-key"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

python -m lmms_eval \
  --model openai \
  --model_args model=moonshotai/kimi-k2.5 \
  --tasks vending_bench2 \
  --limit 2 --batch_size 1
```

### 4.3 Metrics

Agentic tasks report trace-level metrics beyond simple success rate:

| Metric | Description |
|--------|-------------|
| **success** | Binary: did the model reach the target state before submitting? |
| **trace_step_validity** | Fraction of tool calls that executed without error. |
| **trace_state_progress** | How close the final state is to the target (normalized 0-1). |
| **trace_termination_quality** | Did the model emit a proper `<submit>` (vs. hitting the step limit)? |
| **trace_quality** | Average of step validity, state progress, and termination quality. |

### 4.4 YAML Configuration

```yaml
output_type: generate_until_agentic
generation_kwargs:
  max_new_tokens: 256
  temperature: 0
  max_agentic_steps: 10  # loop budget
```

### 4.5 Adding Your Own Agentic Task

1. Create a JSONL dataset with `initial_state`, `target_state`, `tools` (name + description), and `user_query` fields.
2. Implement tool functions in `utils.py` that mutate state deterministically.
3. Write a `doc_to_text` callback that parses `<tool_call>` / `<submit>` tags, calls tools, and returns the 5-tuple.
4. Set `output_type: generate_until_agentic` in the task YAML.
5. Define `process_results` to extract metrics from the JSON payload.

The seed tasks (`vending_bench2`, `tau2_bench`) serve as reference implementations.

---

## 5. Image/Video I/O Throughput Upgrade

This update consolidates image encoding in shared helpers and optimizes video decode hot paths while preserving task-facing semantics:

- **Up to 3.58x faster video decode** (8 frames) and **1.95x at 32 frames** with TorchCodec multi-threaded backend. PyAV is single-threaded by design; TorchCodec parallelizes frame decode across worker threads, so the gap widens as frame counts grow. Set `LMMS_VIDEO_TORCHCODEC_THREADS=8`.
- **2.7x pipeline speedup on LongVideoBench** (decode latency `2.79s` -> `1.02s`, scores unchanged). The unified `read_video` entry point eliminates redundant container open/close cycles that previously happened per-model, and preallocates the output array instead of building a Python list and calling `np.stack`.
- **LRU caching on media-path resolution and image encoding** removes repeated filesystem and base64 work that dominated wall time in large evaluation runs. No regression at sparse sampling (fps=1) — PyAV remains the default.

### 5.1 read_video — Unified Video Decode Entry Point

The `read_video` function in `lmms_eval/models/model_utils/load_video.py` is the single entry point for video frame extraction across all model backends. It uniformly samples `num_frm` frames (or uses FPS-guided sampling when `fps` is set) and returns an `np.ndarray` of shape `(N, H, W, 3)` in `uint8`.

```python [lmms_eval/models/model_utils/load_video.py]
read_video(
    video_path,
    *,
    num_frm=8,
    fps=None,
    format="rgb24",
    force_include_last_frame=False,
    backend=None,
) -> np.ndarray
```

**Supported decode backends** (select via `backend` parameter or `LMMS_VIDEO_DECODE_BACKEND` env var):

| Backend | Install | Default | Notes |
|---------|---------|---------|-------|
| `pyav` | Included (PyAV) | ✅ | Stream-first decode for mp4; packet fallback for webm/mkv. `thread_type="AUTO"` enabled. |
| `torchcodec` | `uv add torchcodec` | | Thread-tunable via `LMMS_VIDEO_TORCHCODEC_THREADS`. See [Section 5.3](#53-torchcodec-thread-tuning-benchmarks). |
| `dali` | `nvidia-dali` (GPU) | | GPU-accelerated decode. Requires `LMMS_VIDEO_DALI_DEVICE=gpu`. |

**Additional I/O optimizations in this release:**

- Shared image encoding helper (`encode_image_to_base64`) across protocol and simple adapters
- Path-metadata keyed image encode cache for repeated path inputs
- PyAV stream fallback: `seek(0)` before packet decode on stream failure
- Set-membership lookup in stream frame selection
- Preallocated output array fill (replaces `list` + `np.stack` path)
- Configurable decord threads via `LMMS_VIDEO_DECORD_THREADS`

### 5.2 Check on Long Video Benchmarks

To validate the optimization, we ran `longvideobench_val_v` with an API provider backed model (OpenRouter, `bytedance-seed/seed-1.6-flash`) under fixed settings (`limit=8`, `max_frames_num=4`, `max_image_size=512`).

This replay benchmarks two things at once:
- **Score reproducibility** across baseline and optimized code paths
- **Video decode latency** in the evaluation pipeline

**Results:**

| Metric | Value |
|--------|-------|
| Aggregate score (baseline / opt-run1 / opt-run2) | `0.5` / `0.5` / `0.5` |
| Decode latency reduction | `2.79s` -> `1.02s` (**-63%**, 2.7x speedup) |
| Opt A/A decode variation | `-3.05%` |
| Baseline vs opt-run1 prediction match | `8/8` |
| Opt-run1 vs opt-run2 prediction match | `6/8` |

Aggregate scores remain unchanged. Per-item drift is consistent with remote-provider nondeterminism.

### 5.3 TorchCodec Thread Tuning Benchmarks

5 warmup + 20 measured iterations per config. TorchCodec `v0.10.0` on PyTorch `2.10.0`. First-frame hash parity verified against PyAV baseline. FPS-guided rows use `num_frm=4096`; `fps=1` yields 54 sampled frames, `fps=30` yields 1639 (near full extraction).

| Scenario | Backend | Threads | Mean (ms) | vs PyAV |
|----------|---------|---------|-----------|---------|
| 8 frames | PyAV | — | 196.64 | baseline |
| | TorchCodec | 0 | 197.41 | +0.39% |
| | TorchCodec | 2 | 121.42 | -38.25% |
| | TorchCodec | 4 | 76.10 | -61.30% |
| | **TorchCodec** | **8** | **54.88** | **-72% (3.58x)** |
| 16 frames | PyAV | — | 188.21 | baseline |
| | TorchCodec | 0 | 469.48 | +149% |
| | TorchCodec | 2 | 283.41 | +51% |
| | **TorchCodec** | **4** | **178.80** | **-5% (1.05x)** |
| | TorchCodec | 8 | 263.44 | +40% |
| 32 frames | PyAV | — | 415.76 | baseline |
| | TorchCodec | 0 | 828.36 | +99% |
| | TorchCodec | 4 | 296.24 | -29% |
| | **TorchCodec** | **8** | **213.45** | **-49% (1.95x)** |
| fps=1 (54 fr) | PyAV | — | 208.65 | baseline |
| | TorchCodec | 0 | 903.26 | +333% |
| | TorchCodec | 8 | 228.49 | +10% |
| | TorchCodec | 16 | 207.43 | -2% (parity) |
| fps=30 (1639 fr) | PyAV | — | 1676.29 | baseline |
| | TorchCodec | 0 | 2406.10 | +44% |
| | **TorchCodec** | **4** | **1272.04** | **-24% (1.32x)** |
| | TorchCodec | 8 | 1282.75 | -23% |

**Takeaways:** Set `LMMS_VIDEO_TORCHCODEC_THREADS=8` as default. At sparse sampling (fps=1) TorchCodec offers no advantage — use PyAV. At dense sampling (fps≥30) use threads=4 for 1.32x. Default threads (0/1) are **never** faster and regress up to +150% at 16 frames.

### 5.4 Media Resolver LRU Caching

The new benchmark tasks rely on `resolve_media_reference` to resolve media paths from local directories, task cache folders, and Hugging Face cache roots. In large runs, this repeats the same path-construction work many times.

v0.7 adds in-process LRU caching for deterministic path-expansion helpers:

- `_candidate_roots_cached(..., maxsize=256)` caches candidate root lists keyed by `cache_dir`, `media_type`, and current environment-derived root values.
- `_extension_variants(..., maxsize=4096)` caches extension-normalized path variants for repeated basename/clip-id lookups.
- `resolve_media_reference` still performs `Path.exists()` checks on every call, so newly downloaded media files are discovered immediately. Only pure path-derivation work is cached.

---

## 6. Lance-Backed Video Mode

v0.7 adds an optional Lance-backed video path for MINERVA so metadata and videos can be distributed through Hugging Face in a compact way — no separate video downloads, no dangling file references.

**Why Lance over Parquet or raw files?** Video evaluation datasets have an inherent tension: lightweight metadata (question text, answer IDs) coexists with heavyweight binary blobs (10-50 MB video files per row). Parquet's row-group architecture forces a trade-off — small row groups waste I/O on scalar columns, large ones blow up memory with video bytes — and its page-level compression means accessing a single video requires decompressing an entire row group. Distributing raw files means managing thousands of loose objects with fragile path references.

Lance resolves this with three design properties documented in [Pace et al., 2025](https://arxiv.org/abs/2504.15247) and the [Lance × Hugging Face integration](https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/):

1. **Blob encoding separates metadata from payload.** Columns tagged with `lance-encoding:blob=true` store raw bytes contiguously per row with only a position index in column metadata. Scanning `video_id` or `video_ext` never touches the multi-GB blob pages. `take_blobs(ids=[row_id])` returns a lazy file-like handle with a single positioned range read — no row-group decompression, no full-table scan.
2. **No row groups.** Lance v2 eliminates row groups entirely ([Lance v2 format](https://lancedb.com/blog/lance-v2/)). Each column picks its own optimal page size independently, so scalar metadata uses compact pages while blob columns use large pages tuned for object-storage I/O. The arXiv paper benchmarks show at most 1 IOP for fixed-width random access and at most 2 IOPs for variable-width, regardless of nesting or compression.
3. **Single-artifact distribution.** One Lance dataset directory on the Hub packages metadata, video bytes, and optional indexes together. Users point to a single `hf://` URI and the resolver streams blobs on demand, caching locally. This replaces the typical "metadata table here, video files there" pattern that requires download scripts and manual path wiring.

### 6.1 Dataset Package on Hugging Face

MINERVA is published as `lmms-lab-eval/minerva` with two key assets:

| Asset | Purpose |
|-------|---------|
| `minerva.json` | Task metadata used by `minerva.yaml` |
| `data/train.lance` | Lance table with one row per `video_id` |

The Lance table stores video bytes in `video_blob` with blob encoding metadata (`lance-encoding:blob=true`) so samples can be fetched by row ID through `take_blobs`. Under the hood, `train.lance` is a directory containing versioned data fragments (each up to 4 GiB), transaction logs, and a protobuf manifest — not a single monolithic file. The blob encoding stores each video's raw bytes contiguously in the data fragments and keeps only a lightweight position-and-length index in the column metadata. This means scanning scalar columns (`video_id`, `video_ext`) never reads blob pages, while `take_blobs(ids=[row_id])` resolves a video in one positioned range read against the data fragment. The `take_blobs` API returns a lazy file-like handle, so the bytes can be piped directly into PyAV or torchcodec without materializing the entire video in memory.

### 6.2 Build Lance Table from Local Downloads

Convert local video files to a Lance table with the provided script:

```bash [Terminal]
uv run --with pylance --with pyarrow python tools/minerva_to_lance.py \
  --metadata-json data/minerva/minerva.json \
  --videos-dir data/minerva/videos \
  --output data/minerva_hf_package/data/train.lance \
  --batch-size 6
```

The resulting Lance schema contains: `video_id`, `youtube_url`, `video_ext`, `video_size_bytes`, `video_blob`.

::note
Install `pylance` (module import name: `lance`) and `pyarrow` for Lance-mode usage.
::

### 6.3 Runtime Resolution Order

At evaluation time, MINERVA resolves videos using this priority:

1. **Local file** via `MINERVA_VIDEO_DIR`
2. **Lance blob** via `MINERVA_LANCE_VIDEO_URI`
3. **YouTube URL** fallback (`https://www.youtube.com/watch?v=<video_id>`)

Configure Lance mode:

```bash [Terminal]
export MINERVA_LANCE_VIDEO_URI="hf://datasets/lmms-lab-eval/minerva/data/train.lance"
export MINERVA_LANCE_VIDEO_ID_COLUMN="video_id"
export MINERVA_LANCE_VIDEO_BLOB_COLUMN="video_blob"
export MINERVA_LANCE_CACHE_DIR="~/.cache/lmms_eval/minerva_lance_videos"
export LANCE_IO_THREADS="64"
export LANCE_CPU_THREADS="8"
```

For local-first mode:

```bash [Terminal]
export MINERVA_VIDEO_DIR="/absolute/path/to/minerva/videos"
```

### 6.4 Run Example

```bash [Terminal]
uv run --with pylance --with pyarrow python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks minerva \
  --batch_size 1 \
  --limit 8
```

MINERVA remains reproducible in two modes: fully local video files or remote Lance blobs from the Hub.

---

## 7. Safety and Red-Teaming Baseline

v0.7 adds a safety/red-teaming task group based on JailbreakBench behaviors. This addresses a gap where lmms-eval had no built-in safety benchmark for jailbreak robustness and over-refusal analysis.

### 7.1 New Task Group

- `safety_redteam` (group)
  - `safety_jailbreakbench_harmful`
  - `safety_jailbreakbench_benign`

Dataset source: `JailbreakBench/JBB-Behaviors` (`behaviors` config, harmful + benign splits).

### 7.2 Reported Metrics

**Harmful split**:

| Metric | Description |
|--------|-------------|
| `jailbreak_asr` | Attack success proxy (lower is better) |
| `refusal_rate` | Refusal behavior under harmful prompts (higher is better) |
| `toxicity_score` | Toxicity estimate in [0, 1] (lower is better) |
| `content_filter_rejection_rate` | Policy/filter rejection frequency (higher is better) |
| `demographic_refusal_rate` | Refusal rate on demographic-related prompts |
| `non_demographic_refusal_rate` | Refusal rate on non-demographic prompts |

**Benign split**:

| Metric | Description |
|--------|-------------|
| `over_refusal_rate` | Benign prompts incorrectly refused (lower is better) |
| `benign_toxicity_score` | Toxicity estimate on benign prompts (lower is better) |
| `content_filter_rejection_rate` | Policy/filter rejection frequency |
| `demographic_refusal_rate` | Refusal rate on demographic-related prompts |
| `non_demographic_refusal_rate` | Refusal rate on non-demographic prompts |

### 7.3 Toxicity Metrics

Toxicity scoring supports two modes:

1. **Perspective API** when `PERSPECTIVE_API_KEY` is configured
2. **Offline keyword heuristic** fallback when the API is unavailable

This keeps safety evaluation usable in both cloud and offline environments.

### 7.4 Usage

```bash [Terminal]
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks safety_redteam \
  --batch_size 1 \
  --limit 20
```

---

## 8. Better One-Line Evaluation Support

Running an evaluation used to mean assembling a long command with many flags. Sharing that command meant copy-pasting a fragile shell one-liner and hoping the environment was set up correctly. Reproducing a result from a paper meant reverse-engineering the setup from a results JSON that stored only a few fields.

v0.7 replaces all of that with a single YAML file:

```bash [Terminal]
python -m lmms_eval --config configs/my_experiment.yaml
```

One file captures everything — model, tasks, generation parameters, and environment variables. Ship the YAML, reproduce the result.

### 8.1 Better YAML system

A config file maps directly to CLI arguments, plus an optional `env` section for environment variables.

For local GPU evaluation:

```yaml [configs/example_local.yaml]
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

```yaml [configs/example_api.yaml]
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

```yaml [configs/example_batch.yaml]
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

### 8.2 Environment Variables

The `env` section sets environment variables before evaluation starts. Credentials and paths live in the config, not in someone's `.bashrc`.

```yaml [configs/example_api.yaml]
env:
  OPENAI_API_KEY: "sk-..."           # Literal value
  HF_TOKEN: "${HF_TOKEN}"            # Expand from shell environment
  HF_HOME: "${HF_HOME:-/data/cache}" # Expand with default fallback
```

Values containing `${VAR}` expand using the current shell environment. The `${VAR:-default}` syntax provides a fallback default. This keeps secrets out of the YAML while documenting which variables the config needs.

::tip
Keys containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` are automatically masked in log output (e.g., `Config env: OPENAI_API_KEY=********`). The actual values are set correctly in `os.environ`.
::

### 8.3 CLI Override Priority

The priority chain is: **defaults < YAML < CLI**. CLI arguments always win.

```bash [Terminal]
# YAML says limit: null, but you want a quick test
python -m lmms_eval --config configs/example_local.yaml --limit 5

# YAML says batch_size: 1, override for faster throughput
python -m lmms_eval --config configs/example_local.yaml --batch_size 4
```

Keep a "canonical" config in the YAML and override individual values at the command line without editing the file. Override detection compares each CLI argument against argparse defaults — only arguments that differ from defaults count as explicit overrides.

### 8.4 Schema Validation

Unknown keys in the YAML now raise an error with the list of valid keys:

```
ValueError: Unknown keys in config file: ['modle', 'taks'].
Valid keys are: ['batch_size', 'config', 'device', 'gen_kwargs', 'limit', 'log_samples', 'model', 'model_args', ...]
```

Previously, typos like `modle` instead of `model` were silently accepted and ignored at runtime, leading to confusing evaluation failures.

### 8.5 Full Experiment Reproducibility

Results JSON now includes `resolved_cli_args` — the complete resolved configuration after merging defaults, YAML, and CLI overrides:

```json [results.json]
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

Reconstruct the exact YAML config from any results file. No more guessing what flags were used.

### 8.6 Web UI Integration

The Web UI supports YAML import and export:

- **Export**: Configure an evaluation in the Web UI, then click "Export YAML" to download the config as a `.yaml` file. Share it with teammates or commit it to your repo.
- **Import**: Click "Import YAML" to load a config file into the Web UI form. Review or tweak a config before running.

The `env` section maps to the Web UI's environment variables field. The round-trip is lossless — export a YAML, import it back, and the form state is identical.

### 8.7 Example Configs

Three example configs ship in `configs/`:

| File | Use Case |
|------|----------|
| `configs/example_local.yaml` | Local GPU evaluation (Qwen2.5-VL) |
| `configs/example_api.yaml` | API model evaluation (OpenAI-compatible) |
| `configs/example_batch.yaml` | Multiple models in a single run |

Use them as templates:

```bash [Terminal]
cp configs/example_local.yaml configs/my_experiment.yaml
# Edit to your needs, then run
python -m lmms_eval --config configs/my_experiment.yaml
```

---

## 9. Efficiency Metrics Coverage

v0.7 improves efficiency observability, but not all metrics are equally available across backends.

### 9.1 Emitted Metrics

v0.7 adds efficiency data at three levels:

| Level | Fields | Scope |
|-------|--------|-------|
| **Per-sample** | `input_tokens`, `output_tokens`, `reasoning_tokens` | Attached to each evaluation record when backend metadata exists |
| **Run-level** | `total_gen_tokens`, `total_elapsed_time`, `avg_speed` | Aggregated across the full evaluation run |
| **Latency breakdown** | TTFT, TPOT | vLLM chat backends only, via native runtime metrics |

::note
TTFT/TPOT parity across every backend is out of scope for v0.7.
::

### 9.2 TTFT/TPOT Coverage by Backend

| Backend Family | TTFT | TPOT | Available Data |
|----------------|------|------|----------------|
| `vllm` chat backends | ✅ | ✅ | Native runtime metrics, logged as `additional_metrics` |
| `sglang` chat backend | ❌ | ❌ | Wall-clock throughput only |
| OpenAI-compatible APIs (`openai`, `async_openai`) | ❌ | ❌ | Token usage + end-to-end latency; no first-token timestamp |
| HuggingFace local generate | ❌ | ❌ | `model.generate()` wall-clock timing only |

> **Important:** TTFT measures request-to-first-token latency. Throughput measures aggregate speed. They answer different questions and should not be treated as interchangeable.

### 9.3 TTFT/TPOT Gaps

| Backend | Blocker | Path to Coverage |
|---------|---------|-----------------|
| OpenAI-compatible APIs | No first-token timestamp in non-streaming responses | Streaming-first instrumentation; measured TTFT would include network and client overhead |
| SGLang | Batch-level timing only | Per-request first-token timing in generation path |
| HuggingFace local | `generate()` does not expose per-token timing | Token streaming or generation callbacks |

### 9.4 Reporting Guidance

- If TTFT/TPOT is critical, use `vllm` backends.
- If using API/SGLang/HF backends, report throughput and token usage. Treat TTFT as unavailable unless custom instrumentation is enabled.
- Keep metric claims backend-qualified (e.g., "TTFT measured on vLLM runtime metrics").

### 9.5 Token-Based Efficiency in Results JSON

With `--log_samples` enabled, v0.7 emits an `efficiency` section in aggregated results:

- `overall.tokens_per_correct_answer`
- `overall.avg_output_tokens_per_sample`
- Per-task breakdown under `efficiency.by_task`

The v0.7 efficiency output is token-based by design. It does not include price-derived cost fields, so metric comparability does not depend on provider-specific pricing tables.

---

## 10. Pipeline-Level Reasoning Tag Stripping

Reasoning models (Qwen3-VL, DeepSeek-R1, QwQ, etc.) emit `<think>...</think>` blocks in their generated text. Without stripping, these tokens pollute scoring on standard benchmarks. Previously, only 6 model files had ad-hoc handling, and vLLM, SGLang, and OpenAI backends had zero protection.

v0.7 moves reasoning tag stripping into the evaluator pipeline itself. It works uniformly across all model backends.

### 10.1 How It Works

Stripping runs in `evaluator.py` **after** the filter pipeline and **before** `process_results()`. Both raw and cleaned outputs are preserved:

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

This design means:

- **Models return raw output** — no model file needs to handle tag stripping.
- **Scoring is clean** — `process_results()` never sees `<think>` tokens.
- **Analysis is preserved** — the raw chain-of-thought remains available in `--log_samples` output.

### 10.2 Usage

Stripping is enabled by default with `<think>...</think>` tags.

::code-group
```bash [Default (stripping enabled)]
python -m lmms_eval --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
    --tasks mme --limit 8 --log_samples
```

```bash [Disable stripping]
python -m lmms_eval --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
    --tasks mme --reasoning_tags none
```

```bash [Custom tag pairs (JSON)]
python -m lmms_eval --model qwen3_vl --tasks mme \
    --reasoning_tags '[["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]'
```
::

### 10.3 Per-Task Override

Tasks can override the CLI setting via the `reasoning_tags` field in their YAML config. Task-level config takes priority over the CLI flag.

```yaml [task_config.yaml]
reasoning_tags: [["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]
```

Set to `none` or `false` to disable stripping for a specific task.

### 10.4 JSONL Log Output Fields

When `--log_samples` is enabled, each JSONL line contains these fields:

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `int` | Index of the document within the dataset split. |
| `input` | `str` | The prompt/context string sent to the model. |
| `target` | `str` | Ground-truth answer from `doc_to_target()`. |
| `resps` | `list` | Raw model output **before** reasoning tag stripping. Preserves `<think>` blocks for chain-of-thought analysis. Omitted when identical to `filtered_resps`. |
| `filtered_resps` | `list` | Model output **after** filter pipeline + reasoning tag stripping. This is what `process_results()` scores. |
| `doc_hash` | `str` | SHA-256 hash of the document JSON for deduplication and cross-run alignment. |
| `<metric>` | `float/int` | Per-sample metric scores (e.g., `exact_match`, `acc`). Keys depend on the task. |

Example record from a reasoning model:

```json [samples.jsonl]
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

::note
`resps` contains the full chain-of-thought and is useful for debugging reasoning behavior. `filtered_resps` is the canonical scored output — use it when computing or verifying metrics. When no stripping occurs, `resps` is omitted if identical to `filtered_resps`.
::

### 10.5 Implementation Details

| File | Change |
|------|--------|
| `lmms_eval/api/reasoning.py` (NEW) | `strip_reasoning_tags()` and `parse_reasoning_tags_config()` |
| `lmms_eval/evaluator.py` | Strip logic before scoring, dual storage in JSONL |
| `lmms_eval/__main__.py` | `--reasoning_tags` CLI argument |
| `lmms_eval/api/task.py` | Per-task `reasoning_tags` field in `TaskConfig` |
| `lmms_eval/api/instance.py` | `raw_filtered_resps` dict for pre-strip preservation |
| 6 model files | Removed ad-hoc `parse_reasoning_model_answer` calls |

---

## 11. Support customized message_format in async_openai

The `async_openai` model backend receives two changes: an internal refactor for maintainability, and a `message_format` parameter that replaces the previous `is_qwen3_vl` flag and the separate `async_openai_qwen3_vl` model class.

### 11.1 message_format Parameter

Different model families behind OpenAI-compatible APIs require different message serialization. Qwen3-VL needs per-frame timestamps prepended to video frames, while the standard OpenAI format sends frames as plain base64 images.

Previously, an `is_qwen3_vl` boolean flag handled this, then a separate model class (`async_openai_qwen3_vl`). Both approaches scale poorly — every new format requires a new flag or a new file + class + registry entry.

v0.7 replaces this with a single `message_format` parameter:

::code-group
```bash [Standard OpenAI format (default)]
python -m lmms_eval --model async_openai \
    --model_args pretrained=gpt-4o,message_format=openai \
    --tasks mme
```

```bash [Qwen3-VL format (per-frame timestamps)]
python -m lmms_eval --model async_openai \
    --model_args pretrained=Qwen/Qwen3-VL-72B,message_format=qwen3_vl \
    --tasks video_mme
```
::

Adding a new format requires only an `elif` in `prepare_messages()` and a corresponding `to_*_messages()` method in `ChatMessages` — no new files or registry changes.

### 11.2 Refactored Concurrency Control

The `generate_until()` method was a single 130-line function with retry logic, adaptive concurrency control, and request scheduling interleaved. v0.7 decomposes it into focused methods:

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

```python [lmms_eval/models/chat/async_openai.py]
async def run():
    pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
    current_concurrency = self._get_initial_concurrency()
    dispatch_order = self._compute_dispatch_order(requests)
    res = await self._run_scheduling_loop(requests, dispatch_order, pbar, current_concurrency)
    pbar.close()
    return res
```

Concurrency tracking state lives in `_AdaptiveConcurrencyTracker` (a dataclass) instead of scattered `nonlocal` variables in nested closures.

---

## 12. Flattened JSONL Log Output

With `--log_samples` enabled, the per-sample JSONL files previously wrote `resps` and `filtered_resps` as doubly-nested lists:

```json
{"resps": [["The answer is cat"]], "filtered_resps": [["cat"]]}
```

The outer list groups multiple `Instance` objects per document (e.g., one per choice in multiple-choice tasks). For the dominant `generate_until` output type, there is always exactly one Instance per document, making the outer list redundant.

v0.7 flattens the outer list at serialization time when it contains only a single element:

```json
{"resps": ["The answer is cat"], "filtered_resps": ["cat"]}
```

### 12.1 When Flattening Applies

| Output Type | Instances per Doc | Before | After |
|-------------|-------------------|--------|-------|
| `generate_until` | 1 | `[["text"]]` | `["text"]` |
| `generate_until_multi_round` | 1 | `[["text"]]` | `["text"]` |
| `loglikelihood` (MCQ) | N (one per choice) | `[["a"], ["b"], ...]` | `[["a"], ["b"], ...]` (unchanged) |

::note
Flattening only removes the outer wrapper when there is exactly one Instance. Multi-choice tasks with multiple Instances per document remain untouched.
::

### 12.2 Deduplication with Flattened Format

The existing dedup logic (omit `resps` when identical to `filtered_resps`) continues to work with the flattened format. After flattening, the two fields are compared directly — if they match, `resps` is omitted from the JSONL record to save space.

### 12.3 Implementation

The flattening happens in `evaluation_tracker.py` during JSONL serialization, not in the evaluator core. In-memory data structures (`logged_samples`) retain the original nested format so existing consumers (wandb logger, logging utilities) continue to work without changes.

---

## 13. Bug Fixes

- Fix image vs video file path detection in `auto_doc_to_messages` fallback
- Align `osworld_g` polygon scoring with osworld-verified annotations
- Harden `mmsi-bench` utils parsing against malformed model responses
- Fix Whisper `world_size` initialization for single-process runtime
- Fix Audio Flamingo 3 parameter handling
- Restore `task_input_specs/redundancy_refactor.yaml` accidentally deleted by test commit

---

## See Also

- [CHANGELOG.md](CHANGELOG.md) — Full change list with PR links for every item
- [docs/external_usage.md](../advanced/external_usage.md) — CLI subcommands and Python library API
- [configs/](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/configs) — Example YAML configs for local, API, and batch evaluation
- [skills/lmms-eval-guide/](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/skills/lmms-eval-guide) — Agent skill files for model, task, and API server workflows

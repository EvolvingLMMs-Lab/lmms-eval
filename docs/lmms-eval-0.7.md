# LMMs-Eval v0.7

v0.7 makes lmms-eval easier to operate and harder to get wrong. The theme is **operational simplicity** for users and **pipeline maturity** for the framework — the evaluator now handles:
- injecting prompts and parsing outputs from reasoning models
- image/video decode optimization
- efficiency metrics

Fewer flags to remember, fewer things that go wrong, and a single file that captures your entire experiment.

## Table of Contents

- [Upgrading from v0.6](#upgrading-from-v06)
- [1. New Benchmark Tasks](#1-new-benchmark-tasks)
- [2. New Models](#2-new-models)
- [3. Image/Video I/O Throughput Upgrade](#3-imagevideo-io-throughput-upgrade)
- [4. Lance-Backed Video Mode](#4-lance-backed-video-mode)
- [5. Safety and Red-Teaming Baseline](#5-safety-and-red-teaming-baseline)
- [6. Efficiency Metrics Coverage](#6-efficiency-metrics-coverage)
- [7. Skill-Based Agent Workflows](#7-skill-based-agent-workflows)
- [8. Better One-Line Evaluation Support](#8-better-one-line-evaluation-support)
- [9. Pipeline-Level Reasoning Tag Stripping](#9-pipeline-level-reasoning-tag-stripping)
- [10. Support customized message_format in async_openai](#10-support-customized-message_format-in-async_openai)
- [11. Flattened JSONL Log Output](#11-flattened-jsonl-log-output)
- [12. Bug Fixes](#12-bug-fixes)

---

## Upgrading from v0.6

::important
Most workflows are backward-compatible. Two model-backend changes require attention before upgrading.
::

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

- **`parse_reasoning_model_answer` removed** from 6 model files. Reasoning tag stripping is now handled at the pipeline level (see [Section 9](#9-pipeline-level-reasoning-tag-stripping)). Remove any direct calls to this function.

### Backward-Compatible Renames

- `read_video_pyav` renamed to `read_video`. A backward-compatible alias is in place — existing code continues to work.

### Deprecations

No additional deprecations in v0.7. The v0.6 deprecation of `doc_to_visual` + `doc_to_text` for API models remains in effect. Use `doc_to_messages` + `ChatMessages` for new integrations.

### CLI Changes

v0.7 introduces a subcommand architecture (`eval`, `tasks`, `models`, `ui`, `serve`, `power`, `version`). The existing `python -m lmms_eval` invocation continues to work as before — subcommands are additive. See [docs/external_usage.md](external_usage.md) for details.

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
| **AGI & agentic** | ARC-AGI-1, ARC-AGI-2, BrowseComp |
| **Audio** | AMI, CN College Listen MCQ, DREAM TTS MCQ, EuroPal ASR, Song Describer |
| **Safety** | JailbreakBench harmful + benign splits (see [Section 5](#5-safety-and-red-teaming-baseline)) |

All tasks are auto-discovered from their YAML configs in `lmms_eval/tasks/`. No manual registration required. Run `python -m lmms_eval --tasks list` to see all available tasks.

---

## 2. New Models

| Model | Description |
|-------|-------------|
| **NanoVLM** | SigLIP2 + MLP projector + Qwen3-0.6B. Chat-style evaluation with async multi-GPU inference via job-queue dispatch. |
| **Async HF model** | Generic async multi-GPU worker backend for HuggingFace model families. Loads replicas on N GPUs with independent worker threads. |

---

## 3. Image/Video I/O Throughput Upgrade

This update consolidates image encoding in shared helpers and optimizes video decode hot paths while preserving task-facing semantics.

### 3.1 read_video — Unified Video Decode Entry Point

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
| `torchcodec` | `uv add torchcodec` | | Thread-tunable via `LMMS_VIDEO_TORCHCODEC_THREADS`. See [Section 3.3](#33-torchcodec-thread-tuning-benchmarks). |
| `dali` | `nvidia-dali` (GPU) | | GPU-accelerated decode. Requires `LMMS_VIDEO_DALI_DEVICE=gpu`. |

**Additional I/O optimizations in this release:**

- Shared image encoding helper (`encode_image_to_base64`) across protocol and simple adapters
- Path-metadata keyed image encode cache for repeated path inputs
- PyAV stream fallback: `seek(0)` before packet decode on stream failure
- Set-membership lookup in stream frame selection
- Preallocated output array fill (replaces `list` + `np.stack` path)
- Configurable decord threads via `LMMS_VIDEO_DECORD_THREADS`

### 3.2 LongVideoBench Check

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

### 3.3 TorchCodec Thread Tuning Benchmarks

TorchCodec with multi-threading delivers significant speedups over PyAV. The table below summarizes recommended settings for common scenarios.

**Recommended settings:**

| Scenario | Threads Setting | Speedup vs PyAV |
|----------|----------------|-----------------|
| 8 frames | `LMMS_VIDEO_TORCHCODEC_THREADS=8` | **3.58x** |
| 16 frames | `LMMS_VIDEO_TORCHCODEC_THREADS=4` | 1.05x |
| 32 frames | `LMMS_VIDEO_TORCHCODEC_THREADS=8` | 1.95x |
| FPS=1 (sparse sampling) | Use PyAV default | — |
| FPS≥30 (dense sampling) | `LMMS_VIDEO_TORCHCODEC_THREADS=4` | 1.32x |

General recommendation: set `LMMS_VIDEO_TORCHCODEC_THREADS=8`. See [Section 3.4](#34-fps-guided-sampling-1-fps-vs-30-fps) for FPS-guided details.

::important
Default threads (0/1) are **not** faster. TorchCodec with `threads=0` or `1` matches or regresses vs PyAV — up to +150% slower at 16 frames. Always set threads explicitly.
::

**Detailed benchmark data** — 5 warmup + 20 measured iterations per config. TorchCodec `v0.10.0` on PyTorch `2.10.0`. First-frame hash parity verified against PyAV baseline.

**8 frames:**

| Backend | Threads | Mean (ms) | vs PyAV |
|---------|---------|-----------|---------|
| PyAV | — | 196.64 | baseline |
| TorchCodec | 0 | 197.41 | +0.39% |
| TorchCodec | 2 | 121.42 | -38.25% |
| TorchCodec | 4 | 76.10 | -61.30% |
| **TorchCodec** | **8** | **54.88** | **-72% (3.58x)** |

**16 frames:**

| Backend | Threads | Mean (ms) | vs PyAV |
|---------|---------|-----------|---------|
| PyAV | — | 188.21 | baseline |
| TorchCodec | 0 | 469.48 | +149% |
| TorchCodec | 2 | 283.41 | +51% |
| **TorchCodec** | **4** | **178.80** | **-5% (1.05x)** |
| TorchCodec | 8 | 263.44 | +40% |

**32 frames:**

| Backend | Threads | Mean (ms) | vs PyAV |
|---------|---------|-----------|---------|
| PyAV | — | 415.76 | baseline |
| TorchCodec | 0 | 828.36 | +99% |
| TorchCodec | 4 | 296.24 | -29% |
| **TorchCodec** | **8** | **213.45** | **-49% (1.95x)** |


### 3.4 FPS-Guided Sampling (1 FPS vs 30 FPS)

Same video as Section 3.3, with `num_frm=4096`. `fps=1` yields 54 sampled frames; `fps=30` yields 1639 (near full extraction).

**Sparse sampling (fps=1, 54 frames):**

| Backend | Threads | Mean (ms) | vs PyAV |
|---------|---------|-----------|---------|
| PyAV | — | 208.65 | baseline |
| TorchCodec | 0 | 903.26 | +333% |
| TorchCodec | 8 | 228.49 | +10% |
| TorchCodec | 16 | 207.43 | -2% (near parity) |

**Dense sampling (fps=30, 1639 frames):**

| Backend | Threads | Mean (ms) | vs PyAV |
|---------|---------|-----------|---------|
| PyAV | — | 1676.29 | baseline |
| TorchCodec | 0 | 2406.10 | +44% |
| **TorchCodec** | **4** | **1272.04** | **-24% (1.32x)** |
| TorchCodec | 8 | 1282.75 | -23% |

::tip
**Sparse sampling (fps=1)**: TorchCodec offers no advantage. PyAV is sufficient.
**Dense sampling (fps≥30)**: Set `LMMS_VIDEO_TORCHCODEC_THREADS=4` for a 1.32x speedup. The backend advantage only materializes when enough frames are decoded to amortize setup overhead.
::

### 3.5 Media Resolver LRU Caching

The new benchmark tasks rely on `resolve_media_reference` to resolve media paths from local directories, task cache folders, and Hugging Face cache roots. In large runs, this repeats the same path-construction work many times.

v0.7 adds in-process LRU caching for deterministic path-expansion helpers:

- `_candidate_roots_cached(..., maxsize=256)` caches candidate root lists keyed by `cache_dir`, `media_type`, and current environment-derived root values.
- `_extension_variants(..., maxsize=4096)` caches extension-normalized path variants for repeated basename/clip-id lookups.
- `resolve_media_reference` still performs `Path.exists()` checks on every call, so newly downloaded media files are discovered immediately. Only pure path-derivation work is cached.

---

## 4. Lance-Backed Video Mode

v0.7 adds an optional Lance-backed video path for MINERVA so metadata and videos can be distributed through Hugging Face in a single reproducible package.

### 4.1 Dataset Package on Hugging Face

MINERVA is published as `lmms-lab-eval/minerva` with two key assets:

| Asset | Purpose |
|-------|---------|
| `minerva.json` | Task metadata used by `minerva.yaml` |
| `data/train.lance` | Lance table with one row per `video_id` |

The Lance table stores video bytes in `video_blob` with blob encoding metadata (`lance-encoding:blob=true`) so samples can be fetched by row ID through `take_blobs`.

### 4.2 Build Lance Table from Local Downloads

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

### 4.3 Runtime Resolution Order

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

### 4.4 Run Example

```bash [Terminal]
uv run --with pylance --with pyarrow python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks minerva \
  --batch_size 1 \
  --limit 8
```

MINERVA remains reproducible in two modes: fully local video files or remote Lance blobs from the Hub.

### 4.5 Benchmarking Video-Resolution Latency

v0.7 includes a direct resolver benchmark for measuring Lance-mode latency:

::code-group
```bash [Lance mode]
uv run python tools/bench_minerva_video_resolution.py \
  --metadata-json data/minerva/minerva.json \
  --mode lance \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 200 \
  --sample-unique-video
```

```bash [Local-file baseline]
uv run python tools/bench_minerva_video_resolution.py \
  --metadata-json data/minerva/minerva.json \
  --mode local \
  --local-video-dir /absolute/path/to/minerva/videos \
  --limit 200
```

```bash [Pipeline comparison (same decode backend)]
uv run python tools/bench_minerva_pipeline_latency.py \
  --local-video-dir /absolute/path/to/minerva/videos \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 100 \
  --batch-size 1 \
  --decode-num-frames 8
```
::

The benchmark reports absolute latency distribution for `minerva_doc_to_visual`: `startup_ms` (Lance resolver init cost), `cold_*` metrics (first pass, cache-miss heavy), and `warm_*` metrics (second pass, cache-hit heavy).

::tip
On local pre-downloaded videos, local raw and Lance modes are often near-parity because decode cost dominates. Lance advantages become clearer in remote/object-storage access, cross-machine reproducibility, and repeated subset evaluation workflows.
::

**Implementation notes:**
- MINERVA Lance resolver avoids eager full-table `video_id` scan at initialization and resolves rows via filtered scan on demand, reducing startup overhead.
- For large-blob dataset builds, use blob-oriented write settings in `tools/minerva_to_lance.py` (smaller rows-per-file and stable storage version).

---

## 5. Safety and Red-Teaming Baseline

v0.7 adds a safety/red-teaming task group based on JailbreakBench behaviors. This addresses a gap where lmms-eval had no built-in safety benchmark for jailbreak robustness and over-refusal analysis.

### 9.1 New Task Group

- `safety_redteam` (group)
  - `safety_jailbreakbench_harmful`
  - `safety_jailbreakbench_benign`

Dataset source: `JailbreakBench/JBB-Behaviors` (`behaviors` config, harmful + benign splits).

### 9.2 Reported Metrics

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

### 9.3 Toxicity Backends

Toxicity scoring supports two modes:

1. **Perspective API** when `PERSPECTIVE_API_KEY` is configured
2. **Offline keyword heuristic** fallback when the API is unavailable

This keeps safety evaluation usable in both cloud and offline environments.

### 9.4 Usage

```bash [Terminal]
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks safety_redteam \
  --batch_size 1 \
  --limit 20
```

---

## 6. Efficiency Metrics Coverage

v0.7 improves efficiency observability, but not all metrics are equally available across backends.

### 10.1 What v0.7 Covers

"Efficiency metrics complete" in v0.7 means:

- Per-sample token counts in evaluation outputs (`input_tokens`, `output_tokens`, `reasoning_tokens` when backend metadata exists)
- Run-level throughput metrics in results (`total_gen_tokens`, `total_elapsed_time`, `avg_speed`)
- vLLM-backed chat paths report TTFT/TPOT through native runtime metrics

::note
TTFT/TPOT parity across every backend is out of scope for v0.7.
::

### 10.2 TTFT/TPOT Coverage Matrix

| Backend Family | TTFT | TPOT | Notes |
|----------------|------|------|-------|
| `vllm` chat backends | ✅ | ✅ | Reads runtime metrics, logs as `additional_metrics` |
| `sglang` chat backend | ❌ | ❌ | Wall-clock throughput only |
| OpenAI-compatible APIs (`openai`, `async_openai`) | ❌ | ❌ | Token usage + end-to-end latency, no first-token timestamp |
| HuggingFace local generate | ❌ | ❌ | `model.generate()` wall-clock timing only |

::important
TTFT measures request-to-first-token latency. Throughput measures aggregate speed. They answer different questions and should not be treated as interchangeable.
::

### 10.3 Extending TTFT Beyond vLLM

Extending TTFT to other backends is feasible but requires backend-specific work:

| Backend | Feasibility | Caveat |
|---------|-------------|--------|
| OpenAI-compatible APIs | Streaming-first instrumentation + first-chunk timestamp | Measured TTFT includes network and client overhead |
| SGLang | Per-request first-token timing in generation path | Batch-level timing alone is not sufficient |
| HuggingFace local | Token streaming/generation callbacks | Default `generate()` does not expose TTFT |

### 10.4 Reporting Guidance

- If TTFT/TPOT is critical, use `vllm` backends.
- If using API/SGLang/HF backends, report throughput and token usage. Treat TTFT as unavailable unless custom instrumentation is enabled.
- Keep metric claims backend-qualified (e.g., "TTFT measured on vLLM runtime metrics").

### 10.5 Token-Based Efficiency in Results JSON

With `--log_samples` enabled, v0.7 emits an `efficiency` section in aggregated results:

- `overall.tokens_per_correct_answer`
- `overall.avg_output_tokens_per_sample`
- Per-task breakdown under `efficiency.by_task`

The v0.7 efficiency output is token-based by design. It does not include price-derived cost fields, so metric comparability does not depend on provider-specific pricing tables.

---

## 7. Skill-Based Agent Workflows

v0.7 standardizes how coding agents learn and orchestrate lmms-eval workflows through the repository skill:

- `skills/lmms-eval-guide/SKILL.md`
- `skills/lmms-eval-guide/references/models.md`
- `skills/lmms-eval-guide/references/tasks.md`
- `skills/lmms-eval-guide/references/api-server.md`
- `skills/lmms-eval-guide/references/workflows.md`

This turns lmms-eval from a set of docs into a reusable operational skill for agents — discover the right integration path, apply correct file-level patterns, and schedule evaluation jobs safely.

### 11.1 Add New Models and Tasks via Skill References

The skill references define recommended implementation paths for extension work:

- **New model integration** -> `references/models.md`
  - Chat-first model template (`is_simple = False`)
  - Request unpacking contract (`request.args` shape)
  - `message_format` parameter for API model serialization dispatch
  - Registration in `lmms_eval/models/__init__.py`
  - v0.6 -> v0.7 breaking changes and migration
  - Minimal verification commands (`--limit 5` / `--verbosity DEBUG`)

- **New task/benchmark integration** -> `references/tasks.md`
  - YAML-first task definition (auto-registered from `tasks/`)
  - `doc_to_messages` + fallback `doc_to_visual` / `doc_to_text`
  - `process_results` + `metric_list` contract
  - `reasoning_tags` per-task override for `<think>` stripping
  - Available task domains (v0.7 coverage)
  - Advanced patterns (`include`, `group`, `cluster_key`, LLM-as-judge)

::tip
Agents should follow this workflow: resolve whether the change is model-side, task-side, or both; load the matching skill reference; follow existing code patterns in nearby models and task YAMLs; run a small-sample verification before broader evaluation.
::

### 11.2 Insert lmms-eval into Training Jobs via HTTP Service

For training-time evaluation, use the eval server workflow from `references/api-server.md`.

Core pattern:

1. Start the HTTP eval server (`launch_server(ServerArgs(...))`) on dedicated eval resources.
2. During training, submit non-blocking jobs with `EvalClient.evaluate(...)`.
3. Continue training immediately. Poll or wait by `job_id` later.
4. Collect metrics asynchronously for checkpoint selection, regression alerts, and model ranking.

Key endpoints for job orchestration:

| Endpoint | Purpose |
|----------|---------|
| `POST /evaluate` | Submit evaluation jobs |
| `GET /jobs/{job_id}` | Query status and results |
| `GET /queue` | Inspect scheduler backlog |
| `GET /tasks` and `GET /models` | Runtime capability discovery |

This service mode is the recommended way to decouple training and evaluation in v0.7 workflows.

### 11.3 HTTP Service as an Operational Primitive

Treat the eval server as infrastructure, not only a convenience API:

- **Queue-safe GPU usage** — scheduler-managed jobs prevent resource contention
- **Async checkpoint evaluation** — evaluate without blocking trainer processes
- **Multi-team reproducibility** — stable request payloads and job IDs
- **Operational visibility** — `/queue` and job lifecycle states

::warning
Run the eval server in trusted environments only. Add authentication, rate limiting, and network isolation before exposing it beyond internal boundaries.
::

### 11.4 Agent Dispatch Strategy

When agents orchestrate lmms-eval tasks, use this routing:

| Scenario | Action |
|----------|--------|
| Task/model extension | Load `lmms-eval-guide` + `references/models.md` or `references/tasks.md` |
| YAML config setup | Load `references/workflows.md` for `--config` usage and config structure |
| Training integration | Load `references/api-server.md` first |
| Quick validation | Run `--limit` smoke tests before full benchmark submission |
| Scalable evaluation | Use HTTP jobs for long-running or periodic evaluation loops |
| Debugging failures | Load `references/workflows.md` for step-by-step debug workflow |

This separates development-time edits (model/task code) from runtime scheduling (HTTP jobs) and operational workflows (config, debugging).

---

## 8. Better One-Line Evaluation Support

Running an evaluation used to mean assembling a long command with many flags. Sharing that command meant copy-pasting a fragile shell one-liner and hoping the environment was set up correctly. Reproducing a result from a paper meant reverse-engineering the setup from a results JSON that stored only a few fields.

v0.7 replaces all of that with a single YAML file:

```bash [Terminal]
python -m lmms_eval --config configs/my_experiment.yaml
```

One file captures everything — model, tasks, generation parameters, and environment variables. Ship the YAML, reproduce the result.

### 5.1 What Goes in the YAML

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

### 5.2 Environment Variables

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

### 5.3 CLI Override Priority

The priority chain is: **defaults < YAML < CLI**. CLI arguments always win.

```bash [Terminal]
# YAML says limit: null, but you want a quick test
python -m lmms_eval --config configs/example_local.yaml --limit 5

# YAML says batch_size: 1, override for faster throughput
python -m lmms_eval --config configs/example_local.yaml --batch_size 4
```

Keep a "canonical" config in the YAML and override individual values at the command line without editing the file. Override detection compares each CLI argument against argparse defaults — only arguments that differ from defaults count as explicit overrides.

### 5.4 Schema Validation

Unknown keys in the YAML now raise an error with the list of valid keys:

```
ValueError: Unknown keys in config file: ['modle', 'taks'].
Valid keys are: ['batch_size', 'config', 'device', 'gen_kwargs', 'limit', 'log_samples', 'model', 'model_args', ...]
```

Previously, typos like `modle` instead of `model` were silently accepted and ignored at runtime, leading to confusing evaluation failures.

### 5.5 Full Experiment Reproducibility

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

### 5.6 Web UI Integration

The Web UI supports YAML import and export:

- **Export**: Configure an evaluation in the Web UI, then click "Export YAML" to download the config as a `.yaml` file. Share it with teammates or commit it to your repo.
- **Import**: Click "Import YAML" to load a config file into the Web UI form. Review or tweak a config before running.

The `env` section maps to the Web UI's environment variables field. The round-trip is lossless — export a YAML, import it back, and the form state is identical.

### 5.7 Example Configs

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

## 9. Pipeline-Level Reasoning Tag Stripping

Reasoning models (Qwen3-VL, DeepSeek-R1, QwQ, etc.) emit `<think>...</think>` blocks in their generated text. Without stripping, these tokens pollute scoring on standard benchmarks. Previously, only 6 model files had ad-hoc handling, and vLLM, SGLang, and OpenAI backends had zero protection.

v0.7 moves reasoning tag stripping into the evaluator pipeline itself. It works uniformly across all model backends.

### 6.1 How It Works

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

### 6.2 Usage

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

### 6.3 Per-Task Override

Tasks can override the CLI setting via the `reasoning_tags` field in their YAML config. Task-level config takes priority over the CLI flag.

```yaml [task_config.yaml]
reasoning_tags: [["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]
```

Set to `none` or `false` to disable stripping for a specific task.

### 6.4 JSONL Log Output Fields

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

### 6.5 Implementation Details

| File | Change |
|------|--------|
| `lmms_eval/api/reasoning.py` (NEW) | `strip_reasoning_tags()` and `parse_reasoning_tags_config()` |
| `lmms_eval/evaluator.py` | Strip logic before scoring, dual storage in JSONL |
| `lmms_eval/__main__.py` | `--reasoning_tags` CLI argument |
| `lmms_eval/api/task.py` | Per-task `reasoning_tags` field in `TaskConfig` |
| `lmms_eval/api/instance.py` | `raw_filtered_resps` dict for pre-strip preservation |
| 6 model files | Removed ad-hoc `parse_reasoning_model_answer` calls |

---

## 10. Support customized message_format in async_openai

The `async_openai` model backend receives two changes: an internal refactor for maintainability, and a `message_format` parameter that replaces the previous `is_qwen3_vl` flag and the separate `async_openai_qwen3_vl` model class.

### 7.1 message_format Parameter

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

### 7.2 Refactored Concurrency Control

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

## 11. Flattened JSONL Log Output

With `--log_samples` enabled, the per-sample JSONL files previously wrote `resps` and `filtered_resps` as doubly-nested lists:

```json
{"resps": [["The answer is cat"]], "filtered_resps": [["cat"]]}
```

The outer list groups multiple `Instance` objects per document (e.g., one per choice in multiple-choice tasks). For the dominant `generate_until` output type, there is always exactly one Instance per document, making the outer list redundant.

v0.7 flattens the outer list at serialization time when it contains only a single element:

```json
{"resps": ["The answer is cat"], "filtered_resps": ["cat"]}
```

### 8.1 When Flattening Applies

| Output Type | Instances per Doc | Before | After |
|-------------|-------------------|--------|-------|
| `generate_until` | 1 | `[["text"]]` | `["text"]` |
| `generate_until_multi_round` | 1 | `[["text"]]` | `["text"]` |
| `loglikelihood` (MCQ) | N (one per choice) | `[["a"], ["b"], ...]` | `[["a"], ["b"], ...]` (unchanged) |

::note
Flattening only removes the outer wrapper when there is exactly one Instance. Multi-choice tasks with multiple Instances per document remain untouched.
::

### 8.2 Deduplication with Flattened Format

The existing dedup logic (omit `resps` when identical to `filtered_resps`) continues to work with the flattened format. After flattening, the two fields are compared directly — if they match, `resps` is omitted from the JSONL record to save space.

### 8.3 Implementation

The flattening happens in `evaluation_tracker.py` during JSONL serialization, not in the evaluator core. In-memory data structures (`logged_samples`) retain the original nested format so existing consumers (wandb logger, logging utilities) continue to work without changes.

---

## 12. Bug Fixes

- Fix image vs video file path detection in `auto_doc_to_messages` fallback
- Align `osworld_g` polygon scoring with osworld-verified annotations
- Harden `mmsi-bench` utils parsing against malformed model responses
- Fix Whisper `world_size` initialization for single-process runtime
- Fix Audio Flamingo 3 parameter handling
- Restore `task_input_specs/redundancy_refactor.yaml` accidentally deleted by test commit

---

## See Also

- [CHANGELOG.md](/CHANGELOG.md) — Full change list with PR links for every item
- [docs/external_usage.md](external_usage.md) — CLI subcommands and Python library API
- [configs/](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/configs) — Example YAML configs for local, API, and batch evaluation
- [skills/lmms-eval-guide/](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/skills/lmms-eval-guide) — Agent skill files for model, task, and API server workflows

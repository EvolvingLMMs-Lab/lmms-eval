<!-- lmms-eval v0.7 -->
# Evaluation Workflows

Covers YAML config-driven evaluation, smoke tests, full runs, reasoning tag stripping, video backends, efficiency metrics, and debug workflows.

## YAML Config-Driven Evaluation (v0.7)

Replace long CLI one-liners with a single YAML file:

```bash
python -m lmms_eval --config configs/my_experiment.yaml
```

### Config Structure

```yaml
# configs/example_local.yaml
env:
  HF_HOME: "${HF_HOME:-~/.cache/huggingface}"
  HF_HUB_ENABLE_HF_TRANSFER: "1"

model: qwen2_5_vl
model_args: "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
tasks: "mme,mmmu_val"
batch_size: 1
seed: "42,42,42,42"
gen_kwargs: "temperature=0,max_new_tokens=1024"
output_path: "results/"
log_samples: true
```

### Key Behaviors

- **Override priority**: defaults < YAML < CLI. CLI always wins.
- **Env expansion**: `${VAR}` expands from shell; `${VAR:-default}` provides fallback.
- **Secret masking**: Keys containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD` are auto-masked in logs.
- **Schema validation**: Unknown YAML keys raise an error listing valid keys (catches typos).
- **Reproducibility**: Results JSON includes `resolved_cli_args` — the full merged config.
- **Batch evaluation**: YAML can be a list of configs for multi-model runs.

### CLI Override Example

```bash
# Use YAML config but override limit for quick test
python -m lmms_eval --config configs/example_local.yaml --limit 5
```

### Example Configs

| File | Use Case |
|------|----------|
| `configs/example_local.yaml` | Local GPU evaluation |
| `configs/example_api.yaml` | API model evaluation (OpenAI-compatible) |
| `configs/example_batch.yaml` | Multiple models in one run |


## Smoke Test Recipes

Always validate with a small sample before scaling up.

```bash
# Local model (Qwen3-VL)
bash examples/models/local_qwen3_vl.sh

# vLLM backend (Qwen3-VL)
bash examples/models/vllm_qwen3_vl.sh

# API backend (OpenRouter/OpenAI-compatible)
bash examples/models/openrouter_api.sh
```

**Quick one-liner** (any model):

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

Pass criteria: metrics print and sample outputs look valid.

## Full Run Recipes

Full runs differ from smoke tests in scale, reproducibility, and output management.

### Key Differences from Smoke Tests

| Aspect | Smoke Test | Full Run |
|--------|-----------|----------|
| `--limit` | `5` or `8` | Omitted or `-1` (full dataset) |
| `--batch_size` | `1` | `8` or `16` (or adaptive for API models) |
| Seed control | Optional | Explicit when needed |
| Output path | Default | Stable naming with optional sample suffixes |

### Cache Strategy

Choose cache flags based on intent:

| Flag | When to Use |
|------|-------------|
| `--use_cache <dir>` | Reuse model responses across runs (e.g., re-scoring with updated metrics) |
| `--cache_requests true` | Speed up request-building (safe for repeated runs with same task/model args) |
| `--cache_requests refresh` | Force rebuild request cache (use after task YAML or model args changed) |
| `--cache_requests delete` | Nuke stale cache entirely (last resort) |

### Example Full Run

```bash
python -m lmms_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=4 \
  --tasks mmmu_val,mme,mathvista \
  --batch_size 8 \
  --log_samples \
  --output_path outputs/qwen25vl-7b
```

## Debug Workflow

Step-by-step approach when something fails.

### 1. Reproduce with Minimal Sample

```bash
python -m lmms_eval \
  --model <model> \
  --model_args <args> \
  --tasks <task> \
  --limit 8 \
  --batch_size 1 \
  --verbosity DEBUG \
  --log_samples
```

### 2. Classify the Failure

Check which pipeline stage fails:

| Stage | Symptom | Where to Look |
|-------|---------|---------------|
| Task loading | `TaskManager` errors, missing YAML keys | Task YAML + `lmms_eval/tasks/__init__.py` |
| Model init | Import errors, `is_simple` mismatch | `lmms_eval/models/__init__.py`, model class |
| Request build | Tuple unpacking errors, wrong arg count | `lmms_eval/api/task.py` `construct_requests` |
| Generation | OOM, empty output, garbled text | Model's `generate_until` method |
| Metric aggregation | KeyError in results, wrong metric shape | Task's `utils.py` `process_results` |

### 3. Targeted Inspection

- **Task bugs**: Check YAML config and `utils.py` contract first. Verify `doc_to_messages` returns correct message structure.
- **Model bugs**: Check `generate_until` tuple unpacking. Chat models expect 5 args, simple models expect 6.
- **Output bugs**: Check `process_results` return keys match `metric_list` in YAML.

### 4. Scale Up Only After Smoke Passes

Once `--limit 8` succeeds, gradually increase to full dataset.

### Useful Diagnostic Commands

```bash
# List all available tasks
uv run python -m lmms_eval --tasks list

# Check model registry
# Inspect lmms_eval/models/__init__.py for AVAILABLE_CHAT_TEMPLATE_MODELS / AVAILABLE_SIMPLE_MODELS

# Lint check
pre-commit run --all-files
```

## Reasoning Tag Stripping (v0.7)

Reasoning models emit `<think>...</think>` blocks. The evaluator strips these automatically.

```bash
# Default: stripping enabled with <think>...</think>
python -m lmms_eval --model qwen3_vl --tasks mme --limit 8 --log_samples

# Disable stripping
python -m lmms_eval --model qwen3_vl --tasks mme --reasoning_tags none

# Custom tag pairs (JSON)
python -m lmms_eval --model qwen3_vl --tasks mme \
    --reasoning_tags '[["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]'
```

Pipeline position: after filter pipeline, before `process_results()`. Raw output preserved in `resps` field; clean output in `filtered_resps`.

## Video Decode Backend Selection

Select via `backend` parameter or `LMMS_VIDEO_DECODE_BACKEND` env var:

| Backend | Install | Default | Best For |
|---------|---------|---------|----------|
| `pyav` | Included | Yes | General use, sparse sampling (fps=1) |
| `torchcodec` | `uv add torchcodec` | | 8-32 frame extraction (up to 3.58x faster with threads=8) |
| `dali` | `nvidia-dali` | | GPU-accelerated dense decode |

```bash
# TorchCodec (MUST set threads explicitly - default is slow)
export LMMS_VIDEO_DECODE_BACKEND=torchcodec
export LMMS_VIDEO_TORCHCODEC_THREADS=8
```

## Efficiency Metrics in Results

With `--log_samples`, v0.7 emits per-sample and run-level efficiency data:

- Per-sample: `input_tokens`, `output_tokens`, `reasoning_tokens`
- Run-level: `total_gen_tokens`, `total_elapsed_time`, `avg_speed`
- Aggregated: `efficiency.overall.tokens_per_correct_answer`, `efficiency.overall.avg_output_tokens_per_sample`, per-task breakdown

TTFT/TPOT available on `vllm` backends only. Other backends report wall-clock throughput.

## Where to Edit (By Intent)

Quick reference for common edit targets:

| Intent | Primary Files | Notes |
|--------|--------------|-------|
| Add/fix a model backend | `models/chat/<name>.py` (preferred), register in `models/__init__.py` | See [models.md](models.md) |
| Add/fix a task | `tasks/<task_name>/<task_name>.yaml` + `utils.py` | See [tasks.md](tasks.md) |
| Debug evaluator logic | `api/task.py`, `evaluator.py`, `evaluator_utils.py` | Request build + execution + metrics |
| Debug reasoning stripping | `api/reasoning.py`, `evaluator.py` | `strip_reasoning_tags()`, `parse_reasoning_tags_config()` |
| Debug output/logging | `loggers/evaluation_tracker.py` | Persistence, JSONL flattening, output shape |
| HTTP eval service | `entrypoints/http_server.py`, `entrypoints/client.py` | See [api-server.md](api-server.md) |
| YAML config loading | `__main__.py` | Config merge logic, schema validation |

<!-- lmms-eval v0.7 -->
# Evaluation Workflows

Smoke tests, full runs, and debug workflows for lmms-eval.

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

## Where to Edit (By Intent)

Quick reference for common edit targets:

| Intent | Primary Files | Notes |
|--------|--------------|-------|
| Add/fix a model backend | `models/chat/<name>.py` (preferred), register in `models/__init__.py` | See [models.md](models.md) |
| Add/fix a task | `tasks/<task_name>/<task_name>.yaml` + `utils.py` | See [tasks.md](tasks.md) |
| Debug evaluator logic | `api/task.py`, `evaluator.py`, `evaluator_utils.py` | Request build + execution + metrics |
| Debug output/logging | `loggers/evaluation_tracker.py` | Persistence and output shape |
| HTTP eval service | `entrypoints/http_server.py`, `entrypoints/client.py` | See [api-server.md](api-server.md) |

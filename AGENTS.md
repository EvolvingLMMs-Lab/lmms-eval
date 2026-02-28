# Agent Guidelines for lmms-eval

This file provides context for AI coding agents (Codex, Devin, SWE-Agent, etc.) working on this codebase.

For detailed development guidelines, see [CLAUDE.md](CLAUDE.md).

## Quick Reference

**Setup**: `uv sync && pre-commit install`
**Run eval**: `python -m lmms_eval --model qwen2_5_vl --tasks mme --limit 5 --batch_size 1`
**Lint**: `pre-commit run --all-files`
**Test registry**: `uv run python -m unittest discover -s test/eval -p "test_model_registry_v2.py"`

### Useful CLI Flags

| Flag | Description |
|------|-------------|
| `--model` | Model backend name (e.g., `qwen2_5_vl`, `openai`, `vllm`) |
| `--model_args` | Comma-separated key=value pairs (e.g., `pretrained=org/model,device_map=auto`) |
| `--tasks` | Comma-separated task names |
| `--limit N` | Only evaluate first N samples (use for quick testing) |
| `--batch_size N` | Batch size for inference |
| `--num_fewshot N` | Number of fewshot examples |
| `--device cuda:0` | Device for local models |
| `--output_path dir/` | Directory for result output |
| `--log_samples` | Save per-sample predictions to output |
| `--verbosity DEBUG` | Set log level (DEBUG, INFO, WARNING, ERROR) |

### Environment Variables

```bash
export OPENAI_API_KEY="..."      # Required for OpenAI/API-backed models
export HF_TOKEN="..."            # Required for gated HuggingFace datasets
export HF_HOME="/path/to/cache"  # HuggingFace cache directory
export HF_HUB_ENABLE_HF_TRANSFER="1"  # Faster downloads
```

## Project Overview

lmms-eval evaluates Large Multimodal Models (LMMs) across image, video, and audio tasks. It supports 100+ benchmarks and 30+ model backends.

### Key Directories

| Path | Purpose |
|------|---------|
| `lmms_eval/models/chat/` | Chat model wrappers (recommended for new models) |
| `lmms_eval/models/simple/` | Legacy model wrappers |
| `lmms_eval/models/__init__.py` | Model registry: `AVAILABLE_SIMPLE_MODELS`, `AVAILABLE_CHAT_TEMPLATE_MODELS`, `MODEL_ALIASES` |
| `lmms_eval/models/registry_v2.py` | `ModelManifest`, `ModelRegistryV2` - aliasing and resolution |
| `lmms_eval/tasks/<task_name>/` | Task configs (YAML) + helper functions (utils.py) |
| `lmms_eval/protocol.py` | `ChatMessages` - structured multimodal message protocol |
| `lmms_eval/api/model.py` | Base class `lmms` - all models subclass this |
| `lmms_eval/api/instance.py` | `Instance` - request object passed to models |
| `lmms_eval/entrypoints/` | HTTP eval server (EvalClient, ServerArgs) |
| `lmms_eval/llm_judge/` | LLM-as-judge scoring providers |

### Model Registry V2

Models are registered via two dicts in `__init__.py` that map `model_id` -> `ClassName`:

- `AVAILABLE_CHAT_TEMPLATE_MODELS` - chat models in `models/chat/`
- `AVAILABLE_SIMPLE_MODELS` - simple models in `models/simple/`

If the same `model_id` exists in both, the registry creates one `ModelManifest` with both paths. Resolution prefers chat over simple.

`MODEL_ALIASES` provides backward-compatible name mappings: `{"new_name": ("old_name_1", "old_name_2")}`.

### Pipeline

```
Dataset --> doc_to_messages (or doc_to_visual + doc_to_text)
        --> Model.generate_until() or Model.loglikelihood()
        --> process_results()
        --> metric aggregation
```

## Common Tasks

### Adding a New Model

1. Create `lmms_eval/models/chat/<name>.py`
2. Subclass `lmms`, set `is_simple = False`, implement `generate_until`
3. Use `@register_model("<name>")` decorator
4. Add `"<name>": "ClassName"` to `AVAILABLE_CHAT_TEMPLATE_MODELS` in `__init__.py`
5. Test: `python -m lmms_eval --model <name> --model_args pretrained=org/model --tasks mme --limit 5`

### Adding a New Task

1. Create `lmms_eval/tasks/<name>/<name>.yaml` + `utils.py`
2. YAML needs: `task`, `dataset_path`, `test_split`, `output_type`, `doc_to_messages`, `process_results`, `metric_list`
3. Tasks auto-register from YAML - no manual registration needed
4. Test: `python -m lmms_eval --model qwen2_5_vl --tasks <name> --limit 8`

#### Task YAML Advanced Features

**`lmms_eval_specific_kwargs`** - Model-specific prompt overrides. Framework selects matching key based on model, falls back to `default`:

```yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer directly."
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."
```

These kwargs are passed to `doc_to_messages(doc, lmms_eval_specific_kwargs=...)`.

**`include`** - Inherit shared config from a template file (avoids duplication across variants):

```yaml
include: _default_template_yaml
```

**`group` + `task` list** - Define task families:

```yaml
group: mmmu
task:
- mmmu_val
- mmmu_test
```

**`output_type`** options: `generate_until` (free-form), `loglikelihood` (multiple-choice), `generate_until_multi_round` (multi-turn conversation)

### Fixing a Model Bug

1. Find the model file in `models/chat/` or `models/simple/`
2. Check `generate_until` for generation issues, `loglikelihood` for multiple-choice
3. Look at `req.args` unpacking - chat models get 5 elements, simple get 6
4. Run with `--limit 5` to verify the fix quickly

### Fixing a Task Bug

1. Task YAML is in `lmms_eval/tasks/<task_name>/`
2. Helper functions are in `utils.py` next to the YAML
3. `process_results` handles scoring, `doc_to_messages` handles input formatting
4. Test with `--limit 8` to verify

## Debugging

### Quick Diagnostics

- **Verbose logging**: `python -m lmms_eval --model ... --verbosity DEBUG` - shows detailed traces
- **Small test run**: `--limit 5` evaluates only 5 samples - always use this when testing changes
- **Log samples**: `--log_samples` saves per-sample predictions to output directory for inspection

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: gen_kwargs['until']` | Wrong type for `until` in generation_kwargs | Must be `str` or `list[str]` |
| `NotImplementedError: loglikelihood` | Model doesn't support multiple-choice | Implement `loglikelihood()` or use `generate_until` tasks only |
| `AttributeError: '_max_length'` | Missing initialization in model `__init__` | Set `self._max_length` in constructor |
| Visual is `None` or `[]` | Dataset sample has no image/video | Guard with `if visual is not None and len(visual) > 0` |
| API timeout/rate limit | API model hitting limits | Use `max_retries` and `retry_backoff_s` in model_args |

### Logging

The codebase uses `eval_logger` from loguru. To add debug logging in your code:

```python
from lmms_eval.utils import eval_logger
eval_logger.debug("Processing batch of {} samples", len(batch))
eval_logger.warning("Missing visual for doc_id={}", doc_id)
```

### Retry Patterns (API Models)

API-backed models (openai, gemini, etc.) support retry configuration:

```bash
python -m lmms_eval --model openai --model_args pretrained=gpt-4o,max_retries=5,retry_backoff_s=2.0 --tasks mme
```

## Constraints

- **Package manager**: uv only, never pip
- **Formatting**: Black (line-length=240) + isort (profile=black). Run `pre-commit run --all-files` before committing.
- **No type suppression**: Never use `as any`, `@ts-ignore`, `type: ignore` to suppress type errors
- **Commits**: Never mention co-authored-by or AI tools
- **Minimal changes**: Fix the specific issue, don't refactor unrelated code
- **Follow patterns**: Match the style of neighboring files exactly

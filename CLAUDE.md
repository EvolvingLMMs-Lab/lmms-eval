# Development Guidelines

This document contains the core development rules for this repository.

## Environment Setup

1. Activate environment first: `source .venv/bin/activate`
2. Initial setup: `uv sync && pre-commit install`
3. After pulling changes: if `uv.lock` changed, run `uv sync`
4. Add dependencies: `uv add <package>`
5. Remove dependencies: `uv remove <package>`
6. Use `uv` only. Never use `pip`.
7. Worktree default: use `.worktrees/<name>` and reuse repo-root `.venv` by symlink.
8. Create per-worktree `.venv` only when isolation is truly required.

For a new worktree:

```bash
ln -s "$(git rev-parse --show-toplevel)/.venv" "$(git rev-parse --show-toplevel)/.worktrees/<name>/.venv"
```

## Formatting and Quality Gate

- Pre-commit hooks: Black (`line-length=240`) and isort (`profile=black`)
- Run before pushing: `pre-commit run --all-files`
- Keep changes minimal and scoped
- Follow existing patterns in neighboring files

## Architecture Essentials

- Entry point: `lmms_eval/__main__.py`
- Base API: `lmms_eval/api/`
- Model registry: `lmms_eval/models/__init__.py` and `lmms_eval/models/registry_v2.py`
- Preferred model type: chat models in `lmms_eval/models/chat/`
- Legacy model type: simple models in `lmms_eval/models/simple/`
- Tasks: `lmms_eval/tasks/<task_name>/` (`*.yaml` + `utils.py`)
- Message protocol: `lmms_eval/protocol.py` (`ChatMessages`)

## Model and Task Workflow

### Add a Model

- Runtime-first default:
  - Prefer `--model vllm` (see `lmms_eval/models/chat/vllm.py`, `lmms_eval/models/chat/vllm_generate.py`)
  - Use `--model sglang` when runtime behavior requires it (`lmms_eval/models/chat/sglang.py`)
  - For API/OpenAI-compatible endpoints, prefer `--model async_openai` (`lmms_eval/models/chat/async_openai.py`)
- Add a dedicated model wrapper only when runtime backends cannot satisfy required behavior.
- Template references:
  - General VLM chat: `lmms_eval/models/chat/qwen2_5_vl.py`, `lmms_eval/models/chat/qwen3_vl.py`
  - HF multimodal wrappers: `lmms_eval/models/chat/internvl_hf.py`, `lmms_eval/models/chat/llava_hf.py`

1. If wrapper is needed, create `lmms_eval/models/chat/<name>.py`
2. Subclass `lmms`, set `is_simple = False`, implement `generate_until`
3. Implement `loglikelihood` when multiple-choice support is required
4. Add `@register_model("<name>")`
5. Register in `AVAILABLE_CHAT_TEMPLATE_MODELS` in `lmms_eval/models/__init__.py`
6. Add `MODEL_ALIASES` when preserving old names
7. Verify with `--limit 5` and the model registry unit test

### Add a Task

1. Create `lmms_eval/tasks/<name>/<name>.yaml` and `utils.py`
2. Use YAML as source of truth (tasks auto-register from YAML)
3. Include core fields: `task`, `dataset_path`, split field, `output_type`, `process_results`, `metric_list`
4. Prefer `doc_to_messages`; use `doc_to_visual`/`doc_to_text` only for legacy simple models
5. Use `lmms_eval_specific_kwargs` for model-specific prompt overrides when needed
6. Use `include` and `group` + `task` for shared templates and task families
7. Verify with `--limit 8`

## Commit and PR Rules

- Keep commit and PR messages focused on problem and fix
- AI attribution and co-author trailers are allowed and encouraged when they reflect actual contribution
- Before commit/push: check `git status` and run pre-commit

## Debugging and Validation Defaults

- Use small runs first: `--limit 5` or `--limit 8`
- Use `--verbosity DEBUG` when diagnosing
- For API-backed models, use retry params in `model_args` (`max_retries`, `retry_backoff_s`)
- Fix CI issues in order: formatting -> type errors -> linting

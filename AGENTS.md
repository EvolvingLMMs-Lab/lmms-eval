# Agent Guidelines for lmms-eval

This file is the fast, repo-specific memory for coding agents.
For fuller development details, see `CLAUDE.md`.

## Quick Start

- Activate environment first: `source .venv/bin/activate`
- Initial setup: `uv sync && pre-commit install`
- Run eval (quick): `python -m lmms_eval --model qwen2_5_vl --tasks mme --limit 5 --batch_size 1`
- Lint all: `pre-commit run --all-files`
- Registry test: `uv run python -m unittest discover -s test/eval -p "test_model_registry_v2.py"`

## Environment and Worktree Policy

- Package manager is `uv` only. Never use `pip`.
- Default shell behavior: run `source .venv/bin/activate` before repo commands.
- Default branch workflow: use git worktrees under `.worktrees/`.
- Default env workflow: reuse repo-root `.venv` for worktrees; do not create per-worktree `.venv` unless isolation is required.
- For a new worktree at `.worktrees/<name>`, link `.venv`:

```bash
ln -s "$(git rev-parse --show-toplevel)/.venv" "$(git rev-parse --show-toplevel)/.worktrees/<name>/.venv"
```

## Architecture Essentials

- Model files:
  - Chat (preferred): `lmms_eval/models/chat/`
  - Simple (legacy): `lmms_eval/models/simple/`
- Model registry: `lmms_eval/models/__init__.py`
  - `AVAILABLE_CHAT_TEMPLATE_MODELS`
  - `AVAILABLE_SIMPLE_MODELS`
  - `MODEL_ALIASES`
- Model resolution details: `lmms_eval/models/registry_v2.py`
- Task definitions: `lmms_eval/tasks/<task_name>/` (`*.yaml` + `utils.py`)
- Task registration: automatic from YAML files.
- Message protocol: `lmms_eval/protocol.py` (`ChatMessages`)

## Common Changes

### Add a New Model (preferred path)

- Runtime-first default (recommended):
  - Use `vllm` runtime first (`--model vllm`; see `lmms_eval/models/chat/vllm.py`, `lmms_eval/models/chat/vllm_generate.py`)
  - Use `sglang` runtime when needed (`lmms_eval/models/chat/sglang.py`)
  - For OpenAI-compatible/API endpoints, use `async_openai` (`lmms_eval/models/chat/async_openai.py`)
- Add a dedicated model wrapper only when runtime backends cannot cover required model behavior.
- Template references by modality:
  - General VLM chat wrapper: `lmms_eval/models/chat/qwen2_5_vl.py`, `lmms_eval/models/chat/qwen3_vl.py`
  - HF multimodal wrappers: `lmms_eval/models/chat/internvl_hf.py`, `lmms_eval/models/chat/llava_hf.py`

1. If wrapper is needed, create `lmms_eval/models/chat/<name>.py`
2. Subclass `lmms`, set `is_simple = False`, implement `generate_until`
3. Implement `loglikelihood` if the model needs multiple-choice support
4. Add `@register_model("<name>")`
5. Register in `AVAILABLE_CHAT_TEMPLATE_MODELS` in `lmms_eval/models/__init__.py`
6. Add `MODEL_ALIASES` entry when preserving old model names
7. Keep request unpacking aligned with model type (`chat`: 5 args, `simple`: 6 args)
8. Quick verify: run eval with `--limit 5` and run registry test

### Add a New Task

1. Create `lmms_eval/tasks/<name>/<name>.yaml` and `utils.py`
2. Required YAML fields: `task`, `dataset_path`, split field (for example `test_split`), `output_type`, `process_results`, `metric_list`
3. Prefer `doc_to_messages` for chat-model-first flow; use `doc_to_visual`/`doc_to_text` only for legacy simple models
4. Use `lmms_eval_specific_kwargs` for model-specific prompt overrides when needed
5. Use `include` to inherit shared task templates and `group` + `task` list for task families
6. Verify locally with `--limit 8` (and OpenRouter smoke test only if relevant)

## Debugging Essentials

- Increase verbosity with `--verbosity DEBUG`
- Keep test runs small (`--limit 5` or `--limit 8`)
- For API-backed models, use retry params in `model_args` (`max_retries`, `retry_backoff_s`)

## Non-Negotiable Constraints

- Keep changes minimal and scoped to the requested issue.
- Match existing code patterns in nearby files.
- Run `pre-commit run --all-files` before pushing.
- AI attribution and co-author trailers are allowed and encouraged when they reflect actual contribution.

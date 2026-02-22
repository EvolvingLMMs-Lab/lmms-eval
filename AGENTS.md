# Agent Guidelines for lmms-eval

This file is the working memory for coding agents in this repository.  
Goal: make safe, minimal, correct changes quickly.

## 1) Quick Start

**Environment**
- Setup: `uv sync && pre-commit install && source .venv/bin/activate`
- Lint/format gate: `pre-commit run --all-files`
- Package manager rule: **use `uv`, never `pip`**

**Fast sanity command**
```bash
bash examples/models/local_qwen3_vl.sh
```

## 2) Codebase Mental Model

lmms-eval runs a simple pipeline:
1. Parse CLI args and config.
2. Resolve tasks through `TaskManager`.
3. Resolve model class via registry (`ModelRegistryV2`), instantiate model.
4. Build per-doc `Instance` requests from task config.
5. Run model methods (`generate_until` / `loglikelihood` / multi-round).
6. Call `process_results`, aggregate metrics.
7. Save aggregated and sample-level outputs.

Key flow anchors:
- CLI: `lmms_eval/__main__.py` (`parse_eval_args`, `cli_evaluate`, `cli_evaluate_single`)
- Evaluator: `lmms_eval/evaluator.py` (`simple_evaluate`, `evaluate`)
- Task indexing/loading: `lmms_eval/tasks/__init__.py` (`TaskManager`, `get_task_dict`)
- Model resolution: `lmms_eval/models/__init__.py`, `lmms_eval/models/registry_v2.py`
- Output writing: `lmms_eval/loggers/evaluation_tracker.py`

## 3) Where To Edit (By Intent)

### Add/fix a model backend
- Implement in `lmms_eval/models/chat/<name>.py` (preferred) or `lmms_eval/models/simple/<name>.py`.
- Register in `lmms_eval/models/__init__.py`:
  - `AVAILABLE_CHAT_TEMPLATE_MODELS` for chat
  - `AVAILABLE_SIMPLE_MODELS` for simple
  - optional aliases in `MODEL_ALIASES`
- Ensure class inherits `lmms_eval.api.model.lmms`.
- `is_simple` contract:
  - chat backend: `is_simple = False`
  - simple backend: `is_simple = True` (default)

### Add/fix a task
- Files live in `lmms_eval/tasks/<task_name>/`:
  - `<task_name>.yaml` for declarative config
  - `utils.py` for `doc_to_messages` / `doc_to_text` / `process_results` / aggregation fns
- Tasks are auto-discovered from YAML. No manual registry update needed.

### Debug evaluator/output logic
- Request build + execution + metric plumbing:
  - `lmms_eval/api/task.py`
  - `lmms_eval/evaluator.py`
  - `lmms_eval/evaluator_utils.py`
- Persistence and output shape:
  - `lmms_eval/loggers/evaluation_tracker.py`

## 4) Model Backend Contract (Important)

`ModelRegistryV2` resolves user model names (including aliases) to a specific chat/simple class path and validates `is_simple`.

Common failure cause: backend type mismatch.
- If resolved as chat but class has `is_simple=True`, load fails.
- If resolved as simple but class has `is_simple=False`, load fails.

### `req.args` shape differs by task/model path
- For `ConfigurableMessagesTask` `generate_until`: `(ctx, doc_to_messages, gen_kwargs, doc_id, task, split)` (6 elements, but second is message builder)
- For `ConfigurableTask` `generate_until`: `(ctx, gen_kwargs, doc_to_visual, doc_id, task, split)` (6 elements)
- For `ConfigurableTask` `loglikelihood`: `(ctx, doc_to_target, doc_to_visual, doc_id, task, split)` (6 elements)

Do not guess tuple order. Confirm in `lmms_eval/api/task.py` `construct_requests`.

## 5) Task YAML Contract

Typical required fields:
- `task`
- `dataset_path`
- split config (usually `test_split` or equivalent)
- `output_type`
- input formatter (`doc_to_messages` preferred; or `doc_to_text` + `doc_to_visual`)
- `process_results`
- `metric_list`

High-value advanced fields:
- `lmms_eval_specific_kwargs`: model-specific prompt variants (default + per-model override)
- `include`: share common template YAML
- `group` + list under `task`: define task families

`output_type` options used in this repo:
- `generate_until`
- `loglikelihood`
- `generate_until_multi_round`
- `generate_until_agentic` (specialized flows)

## 6) Smoke Test Recipes

- **Local model (Qwen3-VL)**: `bash examples/models/local_qwen3_vl.sh`
- **vLLM backend (Qwen3-VL)**: `bash examples/models/vllm_qwen3_vl.sh`
- **API backend (OpenRouter/OpenAI-compatible)**: `bash examples/models/openrouter_api.sh`

The standard smoke test is to run the script and check if it generates metrics and sample outputs successfully. You can also check the saved samples to see if the model generations are valid.

## 7) Full Run Recipes

Smoke tests are for fast validation during development and testing. Full runs are different:
- usually full dataset (`--limit` omitted or `-1`)
- explicit seed control when needed
- larger batch settings for throughput (usually `8` or `16`), or adaptive concurrency control for API-backed models.
- stable output path naming and optional sample suffixes for easier identification.
- cache strategy should follow user intent:
  - use `--use_cache <dir>` only when the user wants response reuse across runs.
  - use `--cache_requests true` to reuse request-building cache for speed.
  - use `--cache_requests refresh` or `--cache_requests delete` only when the user explicitly wants cache invalidation (e.g., task/model args changed and stale cache is not acceptable).

## 8) Recommended Debug Workflow

1. Reproduce with `--limit 8 --verbosity DEBUG --log_samples`.
2. Check whether failure is in task loading, model init, request build, generation, or metric aggregation.
3. Inspect task YAML and `utils.py` contract first for task bugs.
4. Inspect model class `generate_until` and tuple unpacking for model bugs.
5. Only after smoke passes, increase scale.

Useful checks:
- List tasks: `uv run python -m lmms_eval --tasks list`
- Model list/alias checks: inspect `lmms_eval/models/__init__.py`

## 9) Constraints For Agents

- Make minimal, scoped changes. Avoid unrelated refactors.
- Match neighboring style and existing patterns.
- No type-suppression shortcuts (`as any`, `@ts-ignore`, `type: ignore`).
- Run `pre-commit run --all-files` before final handoff when touching code.

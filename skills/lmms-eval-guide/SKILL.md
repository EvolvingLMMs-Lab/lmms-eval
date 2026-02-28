---
name: lmms-eval-guide
version: v0.7
description: Guides AI coding agents through the lmms-eval codebase - a unified evaluation framework for Large Multimodal Models (LMMs). Use when integrating new models, adding evaluation tasks/benchmarks, running YAML config-driven evaluations, orchestrating non-blocking training-time evaluation via the HTTP eval server, or navigating the evaluation pipeline architecture.
---

# lmms-eval Codebase Guide

lmms-eval evaluates Large Multimodal Models across image, video, and audio tasks. 95 model backends (14 chat + 81 simple/legacy), 230 task directories, 1377 YAML configs.

## Scope of This Skill

This file is the **routing layer** for agents. It gives architecture context, decision points, and fast guardrails.

Detailed implementation lives in `references/*.md`.

Use this skill to cover end-to-end workflows:

1. Add or update model backends.
2. Add or update task/benchmark definitions.
3. Run evaluations via YAML configs (`--config`).
4. Insert lmms-eval into training jobs via async API server calls.
5. Operate HTTP eval service (`/evaluate`, `/jobs/{job_id}`, `/queue`, `/tasks`, `/models`).
6. Verify small-sample eval before large-scale runs.
7. Run full-scale evaluations with proper cache, batch, and seed settings.
8. Debug pipeline failures systematically.

## When to Use This Skill

Use this skill when requests include any of the following intents:

- "Add a new model" / "integrate model backend"
- "Add a new task/benchmark" / "write task yaml"
- "Run evaluation with YAML config" / "set up --config"
- "Insert lmms-eval into training job"
- "Run eval asynchronously without blocking training"
- "Start or debug HTTP eval server"
- "Configure reasoning tag stripping" / "<think> tags in output"
- "Run full evaluation" / "production eval run"
- "Debug eval failure" / "eval is broken" / "something failed"
- "Where do I edit X?" / "which file handles Y?"
- "Upgrade from v0.6" / "breaking changes"

## Instructions (Execution Order)

1. Classify the request into one of: model extension, task extension, config setup, training integration, service operations, debugging.
2. Load the matching reference(s) from the routing matrix below.
3. Follow existing patterns in neighboring files before writing new code/config.
4. Validate with a smoke eval (`--limit 5` or `--limit 8`) before broad runs.
5. For YAML config workflows, use `--config` with CLI overrides for quick iteration.
6. For training loops, use async HTTP job submission and collect by `job_id`.
7. For queue or failure issues, inspect `/queue` and `/jobs/{job_id}` and handle terminal states.

## Setup

```bash
uv sync && pre-commit install
# Quick eval test
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks mme --batch_size 1 --limit 8
# Lint
pre-commit run --all-files
```

## Architecture

```
lmms_eval/
├── __main__.py              # CLI entry (python -m lmms_eval)
├── cli/                     # Subcommand dispatch (eval, tasks, models, ui, serve, power, version)
├── evaluator.py             # Core evaluation loop + reasoning tag stripping
├── protocol.py              # ChatMessages - multimodal message protocol
├── api/
│   ├── model.py             # Base class `lmms` - all models subclass this
│   ├── instance.py          # `Instance` - request object passed to models
│   ├── task.py              # ConfigurableTask - task loading from YAML
│   ├── reasoning.py         # strip_reasoning_tags(), parse_reasoning_tags_config()
│   └── registry.py          # @register_model, @register_task decorators
├── models/
│   ├── __init__.py           # AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS, MODEL_ALIASES
│   ├── registry_v2.py        # ModelManifest, ModelRegistryV2 - resolution prefers chat over simple
│   ├── chat/                 # Chat models (14+, RECOMMENDED for new models)
│   │   ├── async_openai.py   # OpenAI-compatible API (message_format parameter)
│   │   └── ...               # qwen2_5_vl, qwen3_vl, vllm, sglang, etc.
│   └── simple/               # Legacy models (81)
├── tasks/                    # Auto-registered from YAML (230 dirs, 1377 configs)
├── entrypoints/              # HTTP eval server (FastAPI)
│   ├── http_server.py        # REST endpoints: /evaluate, /jobs/{job_id}, /queue, /tasks, /models
│   ├── client.py             # EvalClient (sync), AsyncEvalClient
│   └── protocol.py           # EvaluateRequest, JobInfo
├── llm_judge/                # LLM-as-judge scoring
└── tui/                      # Web UI
```

## Evaluation Pipeline

```
CLI --model X --tasks Y  (or --config experiment.yaml)
  1. CONFIG: merge defaults < YAML < CLI overrides -> resolved_cli_args
  2. MODEL: ModelRegistryV2.resolve() -> dynamic import -> cls.create_from_arg_string()
  3. TASKS: scan tasks/ for YAML -> ConfigurableTask(config)
  4. REQUESTS: build Instance objects per doc
     Chat models: (doc_to_messages, gen_kwargs, doc_id, task, split)  # 5 elements
     Simple models: (contexts, gen_kwargs, doc_to_visual, doc_id, task, split)  # 6 elements
  5. INFERENCE: model.generate_until(requests) or model.loglikelihood(requests)
  6. REASONING STRIP: strip_reasoning_tags(filtered_resps) if --reasoning_tags enabled
  7. SCORING: task.process_results(doc, filtered_resps) -> {metric: value}
  8. AGGREGATION: mean, stderr, confidence intervals -> results JSON
```

## ChatMessages Protocol (`protocol.py`)

Structured multimodal message format for chat models:

```python
# Content types: text, image, video, audio
# Message: {"role": "user"|"system"|"assistant", "content": [{type, ...}, ...]}
messages = ChatMessages(messages=raw_messages)
images, videos, audios = messages.extract_media()
hf_messages = messages.to_hf_messages()      # for apply_chat_template()
oai_messages = messages.to_openai_messages()  # for OpenAI API
```

## Model Registration

Models register in `models/__init__.py` via two dicts mapping `model_id -> ClassName`:
- `AVAILABLE_CHAT_TEMPLATE_MODELS` - chat models in `models/chat/`
- `AVAILABLE_SIMPLE_MODELS` - simple models in `models/simple/`

`MODEL_ALIASES` provides backward-compatible name mappings. Registry prefers chat over simple when both exist.

## Reference Routing Matrix

**Load only the reference(s) relevant to your current task.**

| You need to do | Load this reference | Why |
|----------------|---------------------|-----|
| Add a new model backend | [references/models.md](references/models.md) | Model class template, registration, `message_format`, validation |
| Add a new benchmark/task | [references/tasks.md](references/tasks.md) | YAML schema, `reasoning_tags`, `utils.py` contracts, aggregation patterns |
| Start/operate HTTP eval service | [references/api-server.md](references/api-server.md) | Server args, endpoints, client contracts |
| Insert eval into training jobs (non-blocking) | [references/api-server.md](references/api-server.md) | Training-loop integration patterns using async job submission |
| Diagnose queue/job states | [references/api-server.md](references/api-server.md) | Job lifecycle, polling, cancellation, queue inspection |
| Run with YAML config / smoke tests / full evaluations | [references/workflows.md](references/workflows.md) | `--config` usage, smoke test scripts, full run recipes, cache strategy |
| Debug pipeline failures | [references/workflows.md](references/workflows.md) | Step-by-step debug workflow, failure classification, diagnostic commands |
| Find which file to edit | [references/workflows.md](references/workflows.md) | Edit-routing table by intent (model, task, evaluator, output) |

## Multi-Track Requests

For requests spanning multiple tracks (for example, new model + new task + training integration), combine relevant references and execute in this order:

1. Model/task implementation.
2. Smoke validation.
3. HTTP service orchestration.

## Key CLI Flags

| Flag | Description |
|------|-------------|
| `--config` | YAML config file (replaces long CLI one-liners, see [workflows.md](references/workflows.md)) |
| `--model` | Model backend (e.g., `qwen2_5_vl`, `openai`, `vllm`) |
| `--model_args` | Comma key=value pairs (e.g., `pretrained=org/model,device_map=auto`) |
| `--tasks` | Comma-separated task names |
| `--limit N` | Evaluate first N samples only (always use when testing) |
| `--batch_size N` | Batch size for inference |
| `--log_samples` | Save per-sample predictions |
| `--reasoning_tags` | Control `<think>` tag stripping: default `<think>...</think>`, `none` to disable, or custom JSON pairs |
| `--verbosity DEBUG` | Detailed logging |

## Environment Variables

```bash
# Core
export OPENAI_API_KEY="..."                    # API-backed models
export HF_TOKEN="..."                          # Gated HuggingFace datasets
export HF_HOME="/path/to/cache"                # HF cache directory

# Video decode backends
export LMMS_VIDEO_DECODE_BACKEND="pyav"         # pyav (default) | torchcodec | dali
export LMMS_VIDEO_TORCHCODEC_THREADS="8"       # Thread count for TorchCodec (MUST set explicitly)
export LMMS_VIDEO_DALI_DEVICE="gpu"             # GPU decode with DALI

# MINERVA Lance mode
export MINERVA_LANCE_VIDEO_URI="hf://datasets/lmms-lab-eval/minerva/data/train.lance"
export MINERVA_VIDEO_DIR="/path/to/videos"      # Local-first mode

# Safety tasks
export PERSPECTIVE_API_KEY="..."                # Toxicity scoring (optional, falls back to keyword heuristic)
```

## Common Errors

| Error | Fix |
|-------|-----|
| `ValueError: gen_kwargs['until']` | `until` must be `str` or `list[str]` |
| `NotImplementedError: loglikelihood` | Implement `loglikelihood()` or use `generate_until` tasks |
| Visual is `None` or `[]` | Guard: `if visual is not None and len(visual) > 0` |
| `is_simple` mismatch | Set `is_simple = False` on chat model classes |
| `Unknown keys in config file` | Typo in YAML config key. Error message lists valid keys. |
| `<think>` tokens in scored output | Enable `--reasoning_tags` (default) or check per-task `reasoning_tags` override |

## Upgrading from v0.6

| Change | Migration |
|--------|-----------|
| `async_openai_qwen3_vl` model removed | Use `async_openai` with `message_format=qwen3_vl` in `--model_args` |
| `is_qwen3_vl` flag removed | Use `message_format=qwen3_vl` instead |
| `parse_reasoning_model_answer` removed | Reasoning stripping is now pipeline-level via `--reasoning_tags` |
| `read_video_pyav` renamed | Use `read_video`. Backward-compat alias exists. |

## Constraints

- **Package manager**: `uv` only, never `pip`
- **Formatting**: Black (line-length=240) + isort (profile=black). Run `pre-commit run --all-files`
- **Testing**: Always use `--limit 5` or `--limit 8` when testing changes
- **Follow patterns**: Match the style of neighboring files exactly

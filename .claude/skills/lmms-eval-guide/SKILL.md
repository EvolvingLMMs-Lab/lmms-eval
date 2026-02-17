---
name: lmms-eval-guide
version: v0.6
description: Guides AI coding agents through the lmms-eval codebase - a unified evaluation framework for Large Multimodal Models (LMMs). Use when integrating new models, adding evaluation tasks/benchmarks, using the HTTP eval server, or navigating the evaluation pipeline architecture.
---

# lmms-eval Codebase Guide

lmms-eval evaluates Large Multimodal Models across image, video, and audio tasks. 95 model backends (14 chat + 81 simple/legacy), 230 task directories, 1377 YAML configs.

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
├── evaluator.py             # Core evaluation loop
├── protocol.py              # ChatMessages - multimodal message protocol
├── api/
│   ├── model.py             # Base class `lmms` - all models subclass this
│   ├── instance.py          # `Instance` - request object passed to models
│   ├── task.py              # ConfigurableTask - task loading from YAML
│   └── registry.py          # @register_model, @register_task decorators
├── models/
│   ├── __init__.py           # AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS, MODEL_ALIASES
│   ├── registry_v2.py        # ModelManifest, ModelRegistryV2 - resolution prefers chat over simple
│   ├── chat/                 # Chat models (14, RECOMMENDED for new models)
│   └── simple/               # Legacy models (81)
├── tasks/                    # Auto-registered from YAML (230 dirs, 1377 configs)
├── entrypoints/              # HTTP eval server (FastAPI)
│   ├── http_server.py        # REST endpoints: /evaluate, /jobs/{id}, /queue, /tasks, /models
│   ├── client.py             # EvalClient (sync), AsyncEvalClient
│   └── protocol.py           # EvaluateRequest, JobInfo
├── llm_judge/                # LLM-as-judge scoring
└── tui/                      # Web UI
```

## Evaluation Pipeline

```
CLI --model X --tasks Y
  1. MODEL: ModelRegistryV2.resolve() -> dynamic import -> cls.create_from_arg_string()
  2. TASKS: scan tasks/ for YAML -> ConfigurableTask(config)
  3. REQUESTS: build Instance objects per doc
     Chat models: (doc_to_messages, gen_kwargs, doc_id, task, split)  # 5 elements
     Simple models: (contexts, gen_kwargs, doc_to_visual, doc_id, task, split)  # 6 elements
  4. INFERENCE: model.generate_until(requests) or model.loglikelihood(requests)
  5. SCORING: task.process_results(doc, filtered_resps) -> {metric: value}
  6. AGGREGATION: mean, stderr, confidence intervals -> results JSON
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

## When to Load References

| Task | Reference |
|------|-----------|
| Add a new model | [references/models.md](references/models.md) |
| Add a new task/benchmark | [references/tasks.md](references/tasks.md) |
| Use the HTTP eval server | [references/api-server.md](references/api-server.md) |

## Key CLI Flags

| Flag | Description |
|------|-------------|
| `--model` | Model backend (e.g., `qwen2_5_vl`, `openai`, `vllm`) |
| `--model_args` | Comma key=value pairs (e.g., `pretrained=org/model,device_map=auto`) |
| `--tasks` | Comma-separated task names |
| `--limit N` | Evaluate first N samples only (always use when testing) |
| `--batch_size N` | Batch size for inference |
| `--log_samples` | Save per-sample predictions |
| `--verbosity DEBUG` | Detailed logging |

## Environment Variables

```bash
export OPENAI_API_KEY="..."      # API-backed models
export HF_TOKEN="..."            # Gated HuggingFace datasets
export HF_HOME="/path/to/cache"  # HF cache directory
```

## Common Errors

| Error | Fix |
|-------|-----|
| `ValueError: gen_kwargs['until']` | `until` must be `str` or `list[str]` |
| `NotImplementedError: loglikelihood` | Implement `loglikelihood()` or use `generate_until` tasks |
| Visual is `None` or `[]` | Guard: `if visual is not None and len(visual) > 0` |
| `is_simple` mismatch | Set `is_simple = False` on chat model classes |

## Constraints

- **Package manager**: `uv` only, never `pip`
- **Formatting**: Black (line-length=240) + isort (profile=black). Run `pre-commit run --all-files`
- **Testing**: Always use `--limit 5` or `--limit 8` when testing changes
- **Follow patterns**: Match the style of neighboring files exactly

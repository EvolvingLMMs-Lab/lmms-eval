# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Environment Setup

1. **Initial Setup**: `uv sync && pre-commit install`
2. **After Pulling Changes**: If `uv.lock` has changed, run `uv sync` again
3. **Adding Dependencies**: `uv add <package>` (updates both pyproject.toml and uv.lock)
4. **Removing Dependencies**: `uv remove <package>`
5. ONLY use uv, NEVER pip. FORBIDDEN: `uv pip install`, `@latest` syntax

## Code Formatting

Pre-commit hooks run automatically on `git commit` (requires `pre-commit install`):

- **Black**: Python formatting (line-length=240)
- **isort**: Import sorting (profile=black)
- Config: `.pre-commit-config.yaml`
- Run manually: `pre-commit run --all-files`
- Always run before pushing to avoid CI failures

## Code Style

- PEP 8 naming: snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- Type hints required for all code
- Public APIs must have docstrings
- Use f-strings for formatting
- Follow existing patterns exactly

## Architecture Overview (v0.6)

lmms-eval is a unified evaluation framework for Large Multimodal Models supporting image, video, and audio tasks.

### Directory Structure

```
lmms_eval/
├── __main__.py          # CLI entry point
├── api/                 # Base classes: model.lmms, Instance, registry
├── models/
│   ├── __init__.py      # Registry (AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS, MODEL_ALIASES)
│   ├── registry_v2.py   # ModelManifest, ModelRegistryV2 (aliasing + typed resolution)
│   ├── chat/            # Chat models (recommended) - structured ChatMessages input
│   └── simple/          # Simple/legacy models - doc_to_visual + doc_to_text input
├── tasks/               # Task definitions (YAML config + utils.py per task, auto-registered)
├── protocol.py          # ChatMessages protocol (structured multimodal input)
├── entrypoints/         # Eval server (EvalClient, AsyncEvalClient, ServerArgs)
├── llm_judge/           # LLM-as-judge scoring providers
├── loggers/             # Result logging
└── tui/                 # Web UI (React + FastAPI)
```

### Evaluation Pipeline

```
Input (dataset + question) --> Model (inference) --> Judge (metrics)
                                     |
                                 Cache layer (hash of input + config + git_commit)
```

Key v0.6 features:
- **Eval as a Service**: HTTP server for async evaluation during training (`/evaluate`, `/jobs/{id}`, `/queue`)
- **Statistical Rigor**: Confidence intervals, clustered standard errors (`cluster_key`), paired model comparison
- **Model Registry V2**: Single model_id resolves to chat (preferred) or simple. Old names via aliases in `MODEL_ALIASES`.
- **Async Pipeline**: Decoupled inference + judging with cache for crash recovery and deduplication

### Model Types

| Type | Location | Input | Class flag |
|------|----------|-------|------------|
| **Chat** (recommended) | `models/chat/` | `doc_to_messages` -> `ChatMessages` | `is_simple = False` |
| **Simple** (legacy) | `models/simple/` | `doc_to_visual` + `doc_to_text` | `is_simple = True` |

Resolution: chat > simple (unless `force_simple=True`).

### Launch Command

```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,attn_implementation=sdpa --tasks mmmu,mme --batch_size 1 --limit 8 --device cuda:0
```

## How to Add a New Model

### 1. Create model file

```bash
touch lmms_eval/models/chat/my_model.py    # recommended
# or: touch lmms_eval/models/simple/my_model.py  (legacy only)
```

Reference implementations: `chat/qwen2_5_vl.py`, `chat/qwen3_vl.py`, `simple/instructblip.py`.

### 2. Implement the class

Subclass `lmms_eval.api.model.lmms`. Implement `generate_until` (required) and `loglikelihood` (for multiple-choice tasks).

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.protocol import ChatMessages

@register_model("my_model")
class MyModel(lmms):
    is_simple = False  # chat model

    def __init__(self, pretrained: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.model = load_model(pretrained)
        self.processor = load_processor(pretrained)

    def generate_until(self, requests):
        results = []
        for req in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = req.args  # 5 elements for chat
            doc = self.task_dict[task][split][doc_id]
            messages = ChatMessages(messages=doc_to_messages(doc))
            images, videos, audios = messages.extract_media()
            text = self.processor.apply_chat_template(messages.to_hf_messages())
            # ... run inference ...
            results.append(response)
        return results

    def loglikelihood(self, requests):
        # For multiple-choice. Return list[(log_prob, is_greedy)]
        ...
```

### 3. Register in `lmms_eval/models/__init__.py`

```python
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    ...
    "my_model": "MyModel",   # auto-constructs lmms_eval.models.chat.my_model.MyModel
}

# Optional: backward-compatible aliases
MODEL_ALIASES = {
    "my_model": ("old_name",),
}
```

### 4. Test

```bash
python -m lmms_eval --model my_model --model_args pretrained=org/model --tasks mme --limit 5 --batch_size 1
```

## How to Add a New Task

### 1. Create task directory

```
lmms_eval/tasks/my_task/
├── my_task.yaml     # Task config (auto-registered by filename)
└── utils.py         # doc_to_messages, process_results, aggregation functions
```

No manual registration needed - the framework scans `tasks/` for YAML files.

### 2. Write YAML config

```yaml
task: "my_task"
dataset_path: org/my-dataset
dataset_kwargs:
  token: True
test_split: test
output_type: generate_until    # or: loglikelihood, multiple_choice

doc_to_messages: !function utils.my_doc_to_messages  # for chat models (recommended)
doc_to_visual: !function utils.my_doc_to_visual      # for simple models (legacy)
doc_to_text: !function utils.my_doc_to_text          # for simple models (legacy)
doc_to_target: "answer"

generation_kwargs:
  max_new_tokens: 1024
  temperature: 0

process_results: !function utils.my_process_results
metric_list:
  - metric: accuracy
    aggregation: !function utils.my_aggregate
    higher_is_better: true

metadata:
  - version: 0.0
```

### 3. Implement utils.py

```python
def my_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    messages = [{"role": "user", "content": []}]
    if doc.get("image"):
        messages[0]["content"].append({"type": "image", "url": doc["image"]})
    messages[0]["content"].append({"type": "text", "text": doc["question"]})
    return messages

def my_process_results(doc, results):
    pred = results[0].strip()
    return {"accuracy": 1.0 if pred == doc["answer"] else 0.0}

def my_aggregate(results):
    return sum(results) / len(results)
```

### 4. Test

```bash
python -m lmms_eval --model qwen2_5_vl --tasks my_task --limit 8 --batch_size 1
```

## Commit Conventions

- Bug/feature from user report: `git commit --trailer "Reported-by:<name>"`
- Related to GitHub issue: `git commit --trailer "Github-Issue:#<number>"`
- NEVER mention `co-authored-by` or the tool used to create commits or PRs

## Pull Requests

- Focus on the high-level problem description and how it is solved
- Don't go into code specifics unless it adds clarity
- NEVER mention `co-authored-by` or the tool used to create the PR

## Error Resolution

1. CI failures fix order: Formatting -> Type errors -> Linting
2. Run `pre-commit run --all-files` before pushing
3. Check `git status` before commits
4. Keep changes minimal, follow existing patterns
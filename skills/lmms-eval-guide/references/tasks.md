<!-- lmms-eval v0.6 -->
# Adding a New Task/Benchmark

Tasks auto-register from YAML - no manual registration needed. The framework scans `tasks/` for YAML files at startup.

## Directory Structure

```
lmms_eval/tasks/my_task/
├── my_task.yaml              # Task config (auto-registered)
├── _default_template_yaml    # Optional: shared config for variants
└── utils.py                  # doc_to_messages, process_results, aggregation
```

## YAML Config

Create `lmms_eval/tasks/my_task/my_task.yaml`:

```yaml
task: "my_task"
dataset_path: my-org/my-dataset      # HuggingFace dataset path
dataset_kwargs:
  token: True                         # Use HF_TOKEN for gated datasets
test_split: test
output_type: generate_until           # generate_until | loglikelihood | generate_until_multi_round

# Input formatting
doc_to_messages: !function utils.my_doc_to_messages   # Chat models (recommended)
doc_to_visual: !function utils.my_doc_to_visual       # Simple models fallback
doc_to_text: !function utils.my_doc_to_text           # Simple models fallback
doc_to_target: "answer"                               # Field name or !function

# Generation config
generation_kwargs:
  max_new_tokens: 1024
  temperature: 0
  top_p: 1.0
  do_sample: false

# Scoring
process_results: !function utils.my_process_results
metric_list:
  - metric: accuracy
    aggregation: !function utils.my_aggregate
    higher_is_better: true

# Model-specific prompt overrides (framework selects by model name, falls back to "default")
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer with the option's letter from the given choices directly."
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."

metadata:
  - version: 0.0
```

## utils.py Implementation

Create `lmms_eval/tasks/my_task/utils.py`:

```python
from loguru import logger as eval_logger


def my_doc_to_visual(doc):
    """Extract visual content. Used by simple models."""
    if doc.get("image") is not None:
        return [doc["image"].convert("RGB")]
    return []


def my_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format text prompt. Used by simple models."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    question = doc["question"]

    if "options" in doc and doc["options"]:
        options = doc["options"]
        if isinstance(options, str):
            import ast
            options = ast.literal_eval(options)
        choices = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        question = f"{question}\n{choices}"

    return f"{pre}{question}{post}"


def my_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    Structured chat messages for chat models. RECOMMENDED.
    Returns list of message dicts compatible with ChatMessages protocol.
    """
    visuals = my_doc_to_visual(doc)
    text = my_doc_to_text(doc, lmms_eval_specific_kwargs)

    messages = [{"role": "user", "content": []}]
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})
    messages[0]["content"].append({"type": "text", "text": text})
    return messages


def my_process_results(doc, results):
    """
    Score a single sample.
    Args: doc (original document), results (list of model outputs, usually [prediction_string])
    Returns: dict mapping metric_name -> value (keys must match YAML metric_list)
    """
    prediction = results[0].strip()
    answer = str(doc["answer"]).strip()
    return {"accuracy": 1.0 if prediction.lower() == answer.lower() else 0.0}


def my_aggregate(results):
    """Aggregate per-sample scores into final metric."""
    return sum(results) / len(results)
```

## Test

```bash
python -m lmms_eval --model qwen2_5_vl --tasks my_task --limit 8 --batch_size 1
```

## Advanced Features

### Task Groups

Group related tasks for batch evaluation:

```yaml
# lmms_eval/tasks/my_task/my_task.yaml
group: my_task
task:
- my_task_val
- my_task_test
```

### Template Inheritance (`include`)

Share config across variants:

```yaml
# _default_template_yaml
dataset_path: my-org/my-dataset
dataset_kwargs:
  token: True
generation_kwargs:
  max_new_tokens: 1024
  temperature: 0
doc_to_visual: !function utils.my_doc_to_visual
doc_to_text: !function utils.my_doc_to_text
doc_to_messages: !function utils.my_doc_to_messages
process_results: !function utils.my_process_results

# my_task_val.yaml
task: "my_task_val"
test_split: validation
include: _default_template_yaml

# my_task_test.yaml
task: "my_task_test"
test_split: test
include: _default_template_yaml
```

### Video Tasks

```yaml
dataset_path: org/video-dataset
dataset_kwargs:
  token: True
  video: True
cluster_key: videoID    # Correlated questions -> clustered standard errors
```

### LLM-as-Judge Scoring

```python
from lmms_eval.llm_judge import ServerConfig, get_server

server_config = ServerConfig(model_name="gpt-4o-2024-11-20")
server = get_server(server_name="openai", config=server_config)

def my_process_results(doc, results):
    prediction = results[0]
    result = server.evaluate_binary(
        question=doc["question"],
        answer=doc["answer"],
        prediction=prediction,
        output_format="0/1"
    )
    score = int(result["result"]) if result["success"] else 0
    return {"llm_judge_score": score}
```

### Output Types

| Type | Description | Use Case |
|------|-------------|----------|
| `generate_until` | Free-form text generation | Open QA, captioning |
| `loglikelihood` | Multiple-choice via perplexity | MCQ benchmarks |
| `generate_until_multi_round` | Multi-turn conversation | Dialog tasks |

## Real-World Examples

Study these for patterns:

| Task | File | Pattern |
|------|------|---------|
| MME (image QA) | `tasks/mme/mme.yaml` | Basic image task, custom aggregation |
| MMMU (multi-image) | `tasks/mmmu/mmmu_val.yaml` | `doc_to_messages`, template inheritance, model-specific kwargs |
| VideoMME (video) | `tasks/videomme/videomme.yaml` | Video task, `cluster_key` for clustered SE |

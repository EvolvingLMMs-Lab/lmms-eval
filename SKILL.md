---
name: lmms-eval
description: Unified evaluation framework for Large Multimodal Models (LMMs) supporting 100+ benchmarks across image, video, and audio modalities. Use this skill when working with LMM evaluation tasks including running benchmarks, adding new models, creating custom tasks, understanding evaluation results, or extending the framework. Covers CLI usage, SDK interfaces, and integration patterns.
---

# lmms-eval

A standardized evaluation framework for Large Multimodal Models supporting image, video, and audio tasks.

## Quick Start

```bash
# Basic evaluation
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme,mmlu \
  --batch_size 4

# List available tasks
python -m lmms_eval --tasks list
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│                    (__main__.py)                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Evaluator Layer                          │
│              (evaluator.py, simple_evaluate)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│  TaskManager  │ │  Model Layer  │ │    Metrics    │
│  (tasks/)     │ │  (models/)    │ │  (api/)       │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                 │
┌───────▼───────┐ ┌───────▼───────┐
│  YAML Configs │ │ Chat / Simple │
│  + utils.py   │ │   Providers   │
└───────────────┘ └───────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **CLI** | `__main__.py` | Entry point, argument parsing |
| **Evaluator** | `evaluator.py` | Orchestrates evaluation pipeline |
| **Task API** | `api/task.py` | Task interface and configuration |
| **Model API** | `api/model.py` | Model interface (`lmms` base class) |
| **Registry** | `api/registry.py` | Dynamic registration for models/tasks/metrics |
| **Instance** | `api/instance.py` | Request data structure |

### Execution Flow

```
CLI Arguments → TaskManager loads tasks → Build Instance requests
    → Model.generate_until() / loglikelihood() → process_results()
    → Aggregate metrics → Output results
```

## CLI Reference

### Essential Arguments

```bash
python -m lmms_eval [OPTIONS]
```

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model type name | `qwen2_5_vl`, `llava`, `gpt4v` |
| `--model_args` | Model initialization args | `pretrained=path,device=cuda` |
| `--tasks` | Comma-separated task names | `mme,mmlu,mathvista` |
| `--batch_size` | Batch size for evaluation | `4`, `auto` |
| `--output_path` | Results output directory | `./results` |
| `--limit` | Limit samples per task | `100` |
| `--num_fewshot` | Few-shot examples count | `0`, `5` |
| `--log_samples` | Save per-sample outputs | (flag) |

### Advanced Options

```bash
# Generation control
--gen_kwargs temperature=0.7,top_p=0.9,max_new_tokens=256

# Caching
--use_cache ./cache.db
--cache_requests true|refresh|delete

# Configuration file
--config config.yaml

# External tasks
--include_path /path/to/custom/tasks

# Distributed execution (via accelerate)
accelerate launch -m lmms_eval --model llava --tasks mme
```

### Configuration File Format

```yaml
# Single config
model: qwen2_5_vl
model_args: pretrained=Qwen/Qwen2.5-VL-3B-Instruct
tasks: mme,mmlu
batch_size: 4
output_path: ./results

# Multiple configs (list)
- model: llava
  tasks: mme
- model: gpt4v
  tasks: mathvista
```

## SDK Interface

### Programmatic Evaluation

```python
from lmms_eval import evaluator
from lmms_eval.tasks import TaskManager

# Initialize
task_manager = TaskManager()

# Run evaluation
results = evaluator.simple_evaluate(
    model="qwen2_5_vl",
    model_args="pretrained=Qwen/Qwen2.5-VL-3B-Instruct",
    tasks=["mme", "mmlu"],
    batch_size=4,
    limit=100,
    log_samples=True,
)

# Access results
print(results["results"])  # {"mme": {"score": 1850.5}, "mmlu": {"acc": 0.73}}
print(results["samples"])  # Per-sample outputs if log_samples=True
```

### Core API Classes

#### Model Base Class (`api/model.py`)

```python
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from typing import List, Tuple

class lmms:
    """Abstract base class for all models."""

    is_simple: bool = True  # False for chat models

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text for each request."""
        raise NotImplementedError

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Calculate log probability for each request."""
        raise NotImplementedError
```

#### Task Configuration (`api/task.py`)

```python
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict

@dataclass
class TaskConfig:
    # Dataset
    dataset_path: str
    dataset_name: Optional[str] = None
    test_split: str = "test"

    # Input processing
    doc_to_visual: Callable = None    # Extract visual inputs
    doc_to_text: Callable = None      # Extract text prompt
    doc_to_target: str = None         # Target field or function
    doc_to_messages: Callable = None  # Chat format (recommended)

    # Output
    output_type: str = "generate_until"  # or "loglikelihood"
    generation_kwargs: Dict = None

    # Metrics
    process_results: Callable = None
    metric_list: List[Dict] = None

    # Model-specific
    lmms_eval_specific_kwargs: Dict = None
```

#### Instance Structure (`api/instance.py`)

```python
@dataclass
class Instance:
    request_type: str   # "generate_until", "loglikelihood"
    arguments: tuple    # Request-specific args
    idx: int           # Document index
    metadata: dict     # {"task": str, "doc_id": str, "repeats": int}
```

### Registry Decorators

```python
from lmms_eval.api.registry import (
    register_model,
    register_task,
    register_metric,
)

@register_model("my_model", "my_model_alias")
class MyModel(lmms):
    pass

@register_metric(metric="my_metric", higher_is_better=True, aggregation="mean")
def my_metric(predictions, references):
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)
```

## Adding New Models

### Model Types

| Type | Location | Use Case |
|------|----------|----------|
| **Chat** | `models/chat/` | Multimodal messages format (recommended) |
| **Simple** | `models/simple/` | Legacy text + visual separation |

### Chat Model Template

```python
# lmms_eval/models/chat/my_model.py
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from typing import List

@register_model("my_model")
class MyModel(lmms):
    is_simple = False  # Chat model

    def __init__(
        self,
        pretrained: str = "org/model-name",
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self._batch_size = batch_size

        # Load model and processor
        self.model = load_model(pretrained).to(device)
        self.processor = load_processor(pretrained)

    @property
    def batch_size(self):
        return self._batch_size

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        for request in requests:
            # Extract arguments (chat model format)
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            # Get dataset document
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)

            # Process and generate
            inputs = self.processor(messages)
            output = self.model.generate(**inputs, **gen_kwargs)
            text = self.processor.decode(output)

            results.append(text)

        return results

    def loglikelihood(self, requests: List[Instance]) -> List[tuple]:
        # Implement if needed for multiple-choice tasks
        raise NotImplementedError
```

### Register the Model

```python
# lmms_eval/models/__init__.py
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    # ... existing models ...
    "my_model": "MyModel",
}
```

### Simple Model Template

```python
# lmms_eval/models/simple/my_model.py
@register_model("my_simple_model")
class MySimpleModel(lmms):
    is_simple = True  # Default

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        for request in requests:
            # Simple model format: separate visual and text
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)  # List[PIL.Image]
            text_prompt = contexts        # String

            # Process and generate
            output = self.model(images=visuals, prompt=text_prompt)
            results.append(output)

        return results
```

## Adding New Tasks

### Task Structure

```
lmms_eval/tasks/
└── my_task/
    ├── my_task.yaml    # Configuration
    └── utils.py        # Helper functions
```

### Task YAML Configuration

```yaml
# lmms_eval/tasks/my_task/my_task.yaml

# Dataset configuration
dataset_path: huggingface/dataset-name
dataset_name: subset_name  # Optional
test_split: test
validation_split: validation  # Optional

# Input processing (choose one approach)
# Option 1: Chat format (recommended for new tasks)
doc_to_messages: !function utils.doc_to_messages

# Option 2: Simple format (legacy)
doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text

# Target
doc_to_target: answer  # Field name or !function

# Generation settings
output_type: generate_until  # or "loglikelihood", "multiple_choice"
generation_kwargs:
  max_new_tokens: 128
  temperature: 0
  top_p: 1.0
  do_sample: false

# Result processing
process_results: !function utils.process_results

# Metrics
metric_list:
  - metric: accuracy
    aggregation: mean
    higher_is_better: true
  - metric: exact_match
    aggregation: mean
    higher_is_better: true

# Model-specific prompts (optional)
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer:"
  gpt4v:
    pre_prompt: "Image: "
    post_prompt: "\nProvide a brief answer:"
```

### Task Utilities

```python
# lmms_eval/tasks/my_task/utils.py
from PIL import Image
from typing import Dict, List, Any

def doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Extract visual inputs from document."""
    if "image" in doc:
        return [doc["image"].convert("RGB")]
    elif "images" in doc:
        return [img.convert("RGB") for img in doc["images"]]
    return []

def doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Extract text prompt from document."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    return f"{pre}{doc['question']}{post}"

def doc_to_messages(doc: Dict) -> Dict:
    """Convert document to chat messages format (recommended)."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": doc["image"]},
                    {"type": "text", "text": doc["question"]},
                ],
            }
        ]
    }

def process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process model output into metrics."""
    prediction = results[0].strip().lower()
    target = doc["answer"].strip().lower()

    return {
        "accuracy": 1.0 if prediction == target else 0.0,
        "exact_match": 1.0 if results[0].strip() == doc["answer"].strip() else 0.0,
    }
```

### Task Groups

```yaml
# Create task variants or groups
# lmms_eval/tasks/my_task/my_task_variants.yaml

group: my_task_all
task:
  - my_task_easy
  - my_task_hard
  - my_task_expert
```

### Testing Your Task

```bash
# Test with limited samples
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks my_task \
  --limit 5 \
  --log_samples

# Include external task directory
python -m lmms_eval \
  --include_path /path/to/my/tasks \
  --tasks my_custom_task
```

## Metrics

### Built-in Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` / `acc` | Exact match accuracy |
| `exact_match` | String exact match |
| `anls` | Average Normalized Levenshtein Similarity |
| `perplexity` | Model perplexity |

### Custom Metrics

```python
# In task utils.py or register globally
from lmms_eval.api.registry import register_metric

@register_metric(
    metric="f1_score",
    higher_is_better=True,
    aggregation="mean"
)
def f1_score(predictions: List[str], references: List[str]) -> float:
    # Compute F1 score
    pass

# In YAML
metric_list:
  - metric: f1_score
    aggregation: mean
    higher_is_better: true
```

### LLM-as-Judge

For tasks requiring semantic evaluation:

```yaml
# Use LLM judge for evaluation
process_results: !function utils.llm_judge_process

# In utils.py
def llm_judge_process(doc, results):
    from lmms_eval.llm_judge import get_judge_score
    score = get_judge_score(
        question=doc["question"],
        prediction=results[0],
        reference=doc["answer"],
    )
    return {"judge_score": score}
```

## Distributed Evaluation

```bash
# Multi-GPU with accelerate
accelerate launch --num_processes 4 -m lmms_eval \
  --model llava \
  --tasks mme \
  --batch_size 4

# Specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m lmms_eval ...
```

## Caching

```bash
# Enable response caching
python -m lmms_eval \
  --model llava \
  --tasks mme \
  --use_cache ./eval_cache.db

# Cache request building (faster reruns)
--cache_requests true    # Use existing cache
--cache_requests refresh # Rebuild cache
--cache_requests delete  # Clear cache
```

## Output Format

Results are saved to `--output_path`:

```
output_path/
├── results.json          # Aggregated metrics
├── model_outputs.json    # Per-sample outputs (if --log_samples)
└── configs.json          # Run configuration
```

**results.json structure:**
```json
{
  "results": {
    "mme": {
      "mme_perception_score": 1450.5,
      "mme_cognition_score": 320.0,
      "mme_total_score": 1770.5
    },
    "mmlu": {
      "acc": 0.734,
      "acc_stderr": 0.012
    }
  },
  "configs": { ... },
  "versions": { ... }
}
```

## Common Patterns

### Batch Processing in Models

```python
def generate_until(self, requests: List[Instance]) -> List[str]:
    results = []

    # Process in batches
    for i in range(0, len(requests), self.batch_size):
        batch = requests[i:i + self.batch_size]

        # Prepare batch inputs
        batch_inputs = [self._prepare_input(req) for req in batch]

        # Batch inference
        outputs = self.model.generate(batch_inputs)

        results.extend(outputs)

    return results
```

### Handling Multiple Modalities

```python
def doc_to_messages(doc: Dict) -> Dict:
    content = []

    # Add images
    if "images" in doc:
        for img in doc["images"]:
            content.append({"type": "image", "image": img})

    # Add video
    if "video" in doc:
        content.append({"type": "video", "video": doc["video"]})

    # Add audio
    if "audio" in doc:
        content.append({"type": "audio", "audio": doc["audio"]})

    # Add text
    content.append({"type": "text", "text": doc["question"]})

    return {"messages": [{"role": "user", "content": content}]}
```

### Model-Specific Behavior

```yaml
# In task YAML
lmms_eval_specific_kwargs:
  default:
    max_new_tokens: 128
  gpt4v:
    max_new_tokens: 256
    system_prompt: "You are a helpful assistant."
  qwen2_5_vl:
    max_new_tokens: 128
    use_custom_prompt: true
```

```python
# In utils.py
def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    if kwargs.get("use_custom_prompt"):
        return custom_format(doc)
    return default_format(doc)
```

## File Reference

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point |
| `evaluator.py` | Main evaluation orchestration |
| `api/model.py` | Model base class |
| `api/task.py` | Task base class and config |
| `api/instance.py` | Request data structure |
| `api/registry.py` | Registration decorators |
| `api/metrics.py` | Metric functions |
| `models/__init__.py` | Model dynamic loading |
| `tasks/__init__.py` | TaskManager |

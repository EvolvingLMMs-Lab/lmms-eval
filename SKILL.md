# SKILL.md - Comprehensive Codebase Guide for AI Agents

> Deep codebase knowledge for AI coding agents (Claude Code, Codex, Devin, Cursor, SWE-Agent, etc.) working on lmms-eval. Covers architecture, model/task integration patterns, the HTTP eval server API, and the full evaluation pipeline.
>
> For quick-reference commands and conventions, see [AGENTS.md](AGENTS.md).
> For formatting rules and commit conventions, see [CLAUDE.md](CLAUDE.md).

---

## 1. What is lmms-eval?

**lmms-eval** is a unified evaluation framework for Large Multimodal Models (LMMs). It evaluates models that process images, video, and audio across 100+ benchmarks with 30+ model backends.

**Core problem it solves**: Evaluation fragmentation. Benchmarks are scattered across different hosting platforms, each with its own data format, evaluation script, and post-processing logic. Two teams evaluating the same model on the same benchmark with different pipelines get different numbers.

**Design philosophy**: Better evals lead to better models. Forked from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), extended for multimodal with statistical rigor (confidence intervals, clustered standard errors, paired t-test comparison).

**Key stats**: 95 model backends (14 chat + 81 simple), 230 task directories with 1,377 YAML configs, image/video/audio modalities, HTTP eval server for training-time integration.

---

## 2. Architecture Overview

### Directory Structure

```
lmms_eval/
├── __main__.py              # CLI entry point (python -m lmms_eval)
├── evaluator.py             # Core evaluation loop orchestrator
├── protocol.py              # ChatMessages - structured multimodal message protocol
├── api/
│   ├── model.py             # Base class `lmms` - all models subclass this
│   ├── instance.py          # `Instance` - request object passed to models
│   ├── task.py              # Task, ConfigurableTask - task loading from YAML
│   └── registry.py          # @register_model, @register_task decorators, metric registries
├── models/
│   ├── __init__.py           # AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS, MODEL_ALIASES
│   ├── registry_v2.py        # ModelManifest, ModelRegistryV2 - aliasing and typed resolution
│   ├── chat/                 # Chat model wrappers (14 models, RECOMMENDED for new models)
│   │   ├── qwen2_5_vl.py    # Reference: HuggingFace chat model
│   │   ├── qwen3_vl.py      # Reference: video-capable chat model
│   │   ├── openai.py         # Reference: API-based chat model
│   │   ├── vllm.py           # Reference: vLLM inference backend
│   │   ├── sglang.py         # Reference: SGLang inference backend
│   │   └── ...
│   └── simple/               # Legacy model wrappers (81 models)
│       ├── instructblip.py   # Reference: Accelerate-based simple model
│       └── ...
├── tasks/                    # Task definitions (auto-registered from YAML)
│   ├── mme/                  # Example: image QA task
│   │   ├── mme.yaml          # Task config
│   │   └── utils.py          # doc_to_visual, doc_to_text, process_results
│   ├── mmmu/                 # Example: multi-image reasoning with groups
│   │   ├── mmmu.yaml         # Group definition
│   │   ├── mmmu_val.yaml     # Variant with doc_to_messages
│   │   ├── _default_template_yaml  # Template for include directive
│   │   └── utils.py
│   ├── videomme/             # Example: video QA with cluster_key
│   └── ...                   # 230 task directories total
├── entrypoints/              # HTTP evaluation server
│   ├── http_server.py        # FastAPI server with all REST endpoints
│   ├── client.py             # EvalClient (sync) and AsyncEvalClient
│   ├── protocol.py           # Pydantic models: EvaluateRequest, JobInfo, etc.
│   ├── job_scheduler.py      # Sequential GPU-safe job queue
│   └── server_args.py        # ServerArgs configuration
├── llm_judge/                # LLM-as-judge scoring providers
├── loggers/                  # Result logging (wandb, etc.)
└── tui/                      # Web UI (React + FastAPI)
```

### Evaluation Pipeline

```
CLI (python -m lmms_eval --model X --tasks Y)
  │
  ├─ 1. MODEL INSTANTIATION
  │    models.get_model("X") -> ModelRegistryV2.resolve() -> dynamic import -> cls.create_from_arg_string()
  │
  ├─ 2. TASK LOADING
  │    TaskManager.load_task("Y") -> scan tasks/ for YAML -> ConfigurableTask(config)
  │
  ├─ 3. REQUEST BUILDING
  │    task.build_all_requests() -> for each doc: construct Instance objects
  │    Instance.arguments = (doc_to_messages, gen_kwargs, doc_id, task, split)  # chat: 5 elements
  │    Instance.arguments = (contexts, gen_kwargs, doc_to_visual, doc_id, task, split)  # simple: 6 elements
  │
  ├─ 4. MODEL INFERENCE
  │    model.generate_until(requests) or model.loglikelihood(requests)
  │    For each request: unpack args -> call doc_to_messages(doc) -> ChatMessages -> model generates
  │
  ├─ 5. POST-PROCESSING
  │    task.apply_filters() -> extract answers from raw generation
  │    task.process_results(doc, filtered_resps) -> {metric_name: value}
  │
  └─ 6. AGGREGATION
       calculate_aggregate_metric() -> mean, stderr, confidence intervals
       consolidate_group_results() -> aggregate by task groups
       save results JSON + per-sample JSONL
```

### Model Resolution: chat > simple

When a model_id exists in both `AVAILABLE_CHAT_TEMPLATE_MODELS` and `AVAILABLE_SIMPLE_MODELS`, the registry creates one `ModelManifest` with both paths. Resolution prefers chat unless `force_simple=True`.

```python
# Resolution logic in registry_v2.py
if force_simple and manifest.simple_class_path:
    return simple
elif manifest.chat_class_path:
    return chat        # PREFERRED
else:
    return simple      # fallback
```

---

## 3. Quick Reference

### Commands

```bash
# Setup
uv sync && pre-commit install

# Run evaluation
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks mme --batch_size 1 --limit 8

# Lint
pre-commit run --all-files

# Test model registry
uv run python -m unittest discover -s test/eval -p "test_model_registry_v2.py"

# Start HTTP eval server
python -c "from lmms_eval.entrypoints import ServerArgs, launch_server; launch_server(ServerArgs(host='0.0.0.0', port=8000))"

# Start Web UI
uv run lmms-eval-ui
```

### CLI Flags

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

---

## 4. Integrating a New Model

### Step 1: Choose Model Type

| Type | Location | Input Format | When to Use |
|------|----------|-------------|-------------|
| **Chat** (recommended) | `models/chat/` | `doc_to_messages` -> `ChatMessages` | New models, API models, any model with chat template |
| **Simple** (legacy) | `models/simple/` | `doc_to_visual` + `doc_to_text` | Only if model has no chat template support |

### Step 2: Create the Model File

**For chat models** (recommended):

Create `lmms_eval/models/chat/my_model.py`:

```python
from typing import List, Optional, Tuple, Union

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages


@register_model("my_model")
class MyModel(lmms):
    # CRITICAL: Must be False for chat models
    is_simple = False

    def __init__(
        self,
        pretrained: str = "org/my-model",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()

        # Load model and processor
        # Use your model's specific loading logic
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map=device_map,
            torch_dtype="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer

        self.batch_size_per_gpu = int(batch_size)
        self._max_new_tokens = max_new_tokens
        self._device = device

        # Distributed setup
        self._rank = 0
        self._world_size = 1

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        for request in requests:
            # Chat models receive 5 elements
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            # Get the document from task dataset
            doc = self.task_dict[task][split][doc_id]

            # Build structured messages via doc_to_messages callable
            raw_messages = doc_to_messages(doc)
            messages = ChatMessages(messages=raw_messages)

            # Extract media (images, videos, audios)
            images, videos, audios = messages.extract_media()

            # Convert to HuggingFace format and apply chat template
            hf_messages = messages.to_hf_messages()
            text = self.processor.apply_chat_template(
                hf_messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=text,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
            ).to(self._device)

            # Generate
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", self._max_new_tokens),
                temperature=gen_kwargs.get("temperature", 0),
                do_sample=gen_kwargs.get("do_sample", False),
            )

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            response = self.processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
            results.append(response)

        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Required for multiple-choice tasks (loglikelihood output_type)
        # Return list of (log_probability, is_greedy) tuples
        raise NotImplementedError(
            "loglikelihood not implemented for MyModel. "
            "Use generate_until tasks only, or implement this method."
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        # Required for multi-turn conversation tasks
        raise NotImplementedError(
            "generate_until_multi_round not implemented for MyModel."
        )
```

### Step 3: Register in `__init__.py`

Edit `lmms_eval/models/__init__.py`:

```python
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    # ... existing models ...
    "my_model": "MyModel",  # Maps to lmms_eval.models.chat.my_model.MyModel
}

# Optional: backward-compatible aliases
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    # ... existing aliases ...
    "my_model": ("my_model_v1", "old_model_name"),  # Old names still work
}
```

### Step 4: Test

```bash
# Quick test with 5 samples
python -m lmms_eval \
  --model my_model \
  --model_args pretrained=org/my-model \
  --tasks mme \
  --limit 5 \
  --batch_size 1

# Test with verbose logging
python -m lmms_eval \
  --model my_model \
  --model_args pretrained=org/my-model \
  --tasks mme \
  --limit 5 \
  --verbosity DEBUG
```

### Key Implementation Notes

1. **`is_simple` flag**: Must be `False` for chat models, `True` (default) for simple models. Registry validates this at load time.

2. **Request args unpacking**:
   - Chat models: `doc_to_messages, gen_kwargs, doc_id, task, split = request.args` (5 elements)
   - Simple models: `contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args` (6 elements)

3. **`self.task_dict`**: Populated by the framework before inference. Maps `task_name -> split -> doc_id -> document`.

4. **ChatMessages protocol** (`lmms_eval/protocol.py`):
   - `extract_media()` -> returns `(images, videos, audios)` lists
   - `to_hf_messages()` -> HuggingFace format for `apply_chat_template()`
   - `to_openai_messages()` -> OpenAI API format with base64-encoded images

5. **Reference implementations**:
   - HuggingFace local: `chat/qwen2_5_vl.py`, `chat/qwen3_vl.py`
   - API-based: `chat/openai.py`
   - vLLM backend: `chat/vllm.py`
   - SGLang backend: `chat/sglang.py`

---

## 5. Integrating a New Task

### Step 1: Create Task Directory

```
lmms_eval/tasks/my_task/
├── my_task.yaml              # Task config (auto-registered by YAML scanner)
├── _default_template_yaml    # Optional: shared config for task variants
└── utils.py                  # doc_to_messages, process_results, aggregation functions
```

No manual registration needed. The framework scans `tasks/` for YAML files at startup.

### Step 2: Write the YAML Config

Create `lmms_eval/tasks/my_task/my_task.yaml`:

```yaml
task: "my_task"
dataset_path: my-org/my-dataset      # HuggingFace dataset path
dataset_kwargs:
  token: True                         # Use HF_TOKEN for gated datasets
test_split: test                      # Dataset split to evaluate
output_type: generate_until           # generate_until | loglikelihood | generate_until_multi_round

# INPUT FORMATTING
# Modern approach (recommended) - structured messages for chat models
doc_to_messages: !function utils.my_doc_to_messages

# Legacy approach - still needed as fallback for simple models
doc_to_visual: !function utils.my_doc_to_visual
doc_to_text: !function utils.my_doc_to_text
doc_to_target: "answer"               # Field name or !function

# GENERATION CONFIG
generation_kwargs:
  max_new_tokens: 1024
  temperature: 0
  top_p: 1.0
  do_sample: false

# SCORING
process_results: !function utils.my_process_results
metric_list:
  - metric: accuracy
    aggregation: !function utils.my_aggregate
    higher_is_better: true

# MODEL-SPECIFIC PROMPT OVERRIDES
# Framework selects matching key based on model name, falls back to "default"
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer with the option's letter from the given choices directly."
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."
  gpt4v:
    pre_prompt: ""
    post_prompt: "\nAnswer with Yes or No."

metadata:
  - version: 0.0
```

### Step 3: Implement utils.py

Create `lmms_eval/tasks/my_task/utils.py`:

```python
from loguru import logger as eval_logger


# ============================================================================
# INPUT FORMATTING
# ============================================================================

def my_doc_to_visual(doc):
    """Extract visual content from document. Used by simple models."""
    if doc.get("image") is not None:
        return [doc["image"].convert("RGB")]
    return []


def my_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format document into text prompt. Used by simple models."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")

    question = doc["question"]

    # Handle multiple-choice
    if "options" in doc and doc["options"]:
        options = doc["options"]
        if isinstance(options, str):
            import ast
            options = ast.literal_eval(options)
        choices = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )
        question = f"{question}\n{choices}"

    return f"{pre}{question}{post}"


def my_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    Transform document into structured chat messages.
    Used by chat models. RECOMMENDED approach.

    Returns list of message dicts compatible with ChatMessages protocol:
    [{"role": "user", "content": [{"type": "image", "url": ...}, {"type": "text", "text": ...}]}]
    """
    visuals = my_doc_to_visual(doc)
    text = my_doc_to_text(doc, lmms_eval_specific_kwargs)

    messages = [{"role": "user", "content": []}]

    # Add visuals first (images, videos, or audio)
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})

    # Add text prompt
    messages[0]["content"].append({"type": "text", "text": text})

    return messages


# ============================================================================
# RESULT PROCESSING
# ============================================================================

def my_process_results(doc, results):
    """
    Score a single sample.

    Args:
        doc: Original dataset document
        results: List of model outputs (usually [prediction_string])

    Returns:
        Dict mapping metric_name -> metric_value
        Keys must match metric names in YAML metric_list.
    """
    prediction = results[0].strip()
    answer = str(doc["answer"]).strip()

    # Simple exact match
    is_correct = prediction.lower() == answer.lower()

    return {"accuracy": 1.0 if is_correct else 0.0}


def my_aggregate(results):
    """
    Aggregate per-sample scores into final metric.

    Args:
        results: List of values from process_results for this metric

    Returns:
        Single float score
    """
    return sum(results) / len(results)
```

### Step 4: Test

```bash
python -m lmms_eval --model qwen2_5_vl --tasks my_task --limit 8 --batch_size 1
```

### Advanced YAML Features

#### Task Groups

Group related tasks for batch evaluation:

```yaml
# lmms_eval/tasks/my_task/my_task.yaml
group: my_task
task:
- my_task_val
- my_task_test
```

#### Template Inheritance with `include`

Share config across task variants to avoid duplication:

```yaml
# lmms_eval/tasks/my_task/_default_template_yaml
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

# lmms_eval/tasks/my_task/my_task_val.yaml
task: "my_task_val"
test_split: validation
include: _default_template_yaml   # Inherits everything above

# lmms_eval/tasks/my_task/my_task_test.yaml
task: "my_task_test"
test_split: test
include: _default_template_yaml
```

#### Video Tasks

For video evaluation, set `video: True` in dataset_kwargs and use `cluster_key` for correlated questions:

```yaml
dataset_path: org/video-dataset
dataset_kwargs:
  token: True
  video: True
cluster_key: videoID    # Questions from same video are correlated (clustered SE)
```

#### LLM-as-Judge Scoring

For tasks that need GPT-based evaluation:

```python
# In utils.py
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

### Real-World YAML Examples

**MME** (image QA, simple format):
```yaml
dataset_path: lmms-lab/MME
task: "mme"
test_split: test
output_type: generate_until
doc_to_visual: !function utils.mme_doc_to_visual
doc_to_text: !function utils.mme_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  max_new_tokens: 16
  temperature: 0
process_results: !function utils.mme_process_results
metric_list:
  - metric: mme_perception_score
    aggregation: !function utils.mme_aggregate_results
    higher_is_better: true
  - metric: mme_cognition_score
    aggregation: !function utils.mme_aggregate_results
    higher_is_better: true
```

**MMMU** (multi-image reasoning, with doc_to_messages):
```yaml
dataset_path: lmms-lab/MMMU
task: "mmmu_val"
test_split: validation
output_type: generate_until
doc_to_visual: !function utils.mmmu_doc_to_visual
doc_to_text: !function utils.mmmu_doc_to_text
doc_to_messages: !function utils.mmmu_doc_to_messages
doc_to_target: "answer"
process_results: !function utils.mmmu_process_results
metric_list:
  - metric: mmmu_acc
    aggregation: !function utils.mmmu_aggregate_results
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    prompt_type: "format"
    multiple_choice_prompt: "Answer with the option's letter from the given choices directly."
    open_ended_prompt: "Answer the question using a single word or phrase."
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."
include: _default_template_yaml
```

**VideoMME** (video QA with clustered standard errors):
```yaml
dataset_path: lmms-lab/Video-MME
dataset_kwargs:
  token: True
  cache_dir: videomme
  video: True
task: videomme
test_split: test
output_type: generate_until
cluster_key: videoID
doc_to_visual: !function utils.videomme_doc_to_visual
doc_to_text: !function utils.videomme_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  max_new_tokens: 16
  temperature: 0
process_results: !function utils.videomme_process_results
metric_list:
  - metric: videomme_perception_score
    aggregation: !function utils.videomme_aggregate_results
    higher_is_better: true
```

---

## 6. HTTP Evaluation Server (Eval-as-a-Service)

The eval server decouples evaluation from training. Submit async evaluation jobs from your training loop without blocking.

### Server Setup

```python
from lmms_eval.entrypoints import ServerArgs, launch_server

args = ServerArgs(
    host="0.0.0.0",
    port=8000,
    max_completed_jobs=200,      # Max finished jobs kept in memory
    temp_dir_prefix="lmms_eval_" # Prefix for temp directories
)
launch_server(args)  # Blocks, serves at http://0.0.0.0:8000
# API docs auto-generated at http://0.0.0.0:8000/docs
```

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/evaluate` | POST | Submit evaluation job |
| `/jobs/{job_id}` | GET | Get job status and results |
| `/queue` | GET | View queue status |
| `/tasks` | GET | List available tasks |
| `/models` | GET | List available models |
| `/jobs/{job_id}` | DELETE | Cancel queued job |
| `/merge` | POST | Merge FSDP2 sharded checkpoints |

### Python Client Usage

**Synchronous:**

```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")

# Submit evaluation (non-blocking, returns immediately)
job = client.evaluate(
    model="qwen2_5_vl",
    tasks=["mmmu_val", "mme"],
    model_args={"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"},
    batch_size=1,
    device="cuda:0",
)
print(f"Job submitted: {job['job_id']}")

# Poll for completion
result = client.wait_for_job(job["job_id"], poll_interval=5.0, timeout=3600.0)
print(result["result"])
```

**Asynchronous:**

```python
from lmms_eval.entrypoints import AsyncEvalClient

async with AsyncEvalClient("http://eval-server:8000") as client:
    job = await client.evaluate(
        model="qwen3_vl",
        tasks=["mmmu_val"],
        model_args={"pretrained": "Qwen/Qwen3-VL-4B-Instruct"},
    )
    result = await client.wait_for_job(job["job_id"])
```

### Training Loop Integration

```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")
eval_jobs = []

for epoch in range(num_epochs):
    train_one_epoch()

    # Evaluate every 5 epochs (non-blocking)
    if epoch % 5 == 0:
        job = client.evaluate(
            model="vllm",
            model_args={"model": f"checkpoints/epoch_{epoch}"},
            tasks=["mmmu_val", "mathvista"],
        )
        eval_jobs.append((epoch, job["job_id"]))

# After training, collect all results
for epoch, job_id in eval_jobs:
    result = client.wait_for_job(job_id)
    print(f"Epoch {epoch}: {result['result']}")
```

### EvaluateRequest Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model backend name |
| `tasks` | list[string] | Yes | Task names to evaluate |
| `model_args` | dict | No | Model-specific arguments |
| `num_fewshot` | int | No | Few-shot examples (default: 0) |
| `batch_size` | int/string | No | Batch size for inference |
| `device` | string | No | Device (e.g., "cuda:0") |
| `limit` | int/float | No | Limit samples (for testing) |
| `log_samples` | bool | No | Log per-sample outputs (default: true) |
| `num_gpus` | int | No | Number of GPUs (default: 1) |

### Job Status Values

`queued` -> `running` -> `completed` | `failed` | `cancelled`

### Security Note

The server is intended for trusted environments only. Do NOT expose to untrusted networks without authentication, rate limiting, and network isolation.

---

## 7. Core Data Structures

### ChatMessages Protocol (`lmms_eval/protocol.py`)

The structured multimodal message format used by chat models:

```python
# Content types
ChatTextContent:   {"type": "text",  "text": str}
ChatImageContent:  {"type": "image", "url": Any}   # PIL.Image, path, or bytes
ChatVideoContent:  {"type": "video", "url": Any}   # Video path or bytes
ChatAudioContent:  {"type": "audio", "url": Any}   # Audio path or bytes

# Message structure
ChatMessage:  {"role": "user"|"system"|"assistant", "content": [ChatContent, ...]}

# Container with conversion methods
ChatMessages:
    messages: List[ChatMessage]
    extract_media()       -> (images, videos, audios)  # Separate media from messages
    to_hf_messages()      -> HuggingFace format for apply_chat_template()
    to_openai_messages()  -> OpenAI API format with base64-encoded images
```

### Instance (`lmms_eval/api/instance.py`)

The request object passed to model methods:

```python
@dataclass
class Instance:
    request_type: "loglikelihood" | "generate_until" | "generate_until_multi_round"
    arguments: tuple       # Packed args, unpacked via .args property
    idx: int               # Instance index
    metadata: dict         # {"task": name, "doc_id": id, "repeats": n}
    resps: list            # Raw model responses (populated after inference)
    filtered_resps: dict   # Filtered responses (populated after apply_filters)
    task_name: str         # Unpacked from metadata
    doc_id: str            # Unpacked from metadata
```

### ModelManifest (`lmms_eval/models/registry_v2.py`)

Declarative model registration:

```python
@dataclass(frozen=True)
class ModelManifest:
    model_id: str                       # Canonical ID (e.g., "qwen2_5_vl")
    simple_class_path: str | None       # e.g., "lmms_eval.models.simple.qwen2_5_vl.Qwen2_5_VL"
    chat_class_path: str | None         # e.g., "lmms_eval.models.chat.qwen2_5_vl.Qwen2_5_VL"
    aliases: tuple[str, ...] = ()       # Backward-compatible names
```

### Base Model Class (`lmms_eval/api/model.py`)

```python
class lmms(abc.ABC):
    is_simple: bool = True              # False for chat models

    # Abstract methods (must implement)
    def generate_until(self, requests: List[Instance]) -> List[str]: ...
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]: ...
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]: ...

    # Factory method (used by CLI)
    @classmethod
    def create_from_arg_string(cls, arg_string: str, additional_config: dict) -> "lmms":
        # Parses "key1=val1,key2=val2" and instantiates class
        ...
```

---

## 8. Advanced Features

### Response Caching

Enable caching to skip inference on repeated evaluations:

```bash
export LMMS_EVAL_USE_CACHE=True
```

Cached responses are stored in `~/.cache/lmms-eval/eval_cache/` as JSONL files keyed by `(task_name, rank, world_size)`. Cache keys include the full request arguments so different prompts produce different cache entries.

### Distributed Evaluation

```bash
# Multi-GPU with accelerate
accelerate launch --num_processes 4 -m lmms_eval --model qwen2_5_vl --tasks mme

# With vLLM tensor parallelism
python -m lmms_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=4 --tasks mme
```

### Statistical Rigor (v0.6)

- **Confidence intervals**: Bootstrap resampling (100,000 iterations) for robust CI estimation
- **Clustered standard errors**: Use `cluster_key` in YAML when questions are correlated (e.g., same video)
- **Paired comparison**: t-test for comparing two models on the same benchmark

### LLM-as-Judge

The `lmms_eval/llm_judge/` module provides configurable judge backends:

```python
from lmms_eval.llm_judge import ServerConfig, get_server

# Configure judge
config = ServerConfig(model_name="gpt-4o-2024-11-20")
server = get_server(server_name="openai", config=config)

# Binary evaluation (correct/incorrect)
result = server.evaluate_binary(
    question="What color is the sky?",
    answer="blue",
    prediction="The sky appears blue",
    output_format="0/1"
)
# result = {"success": True, "result": "1"}
```

### Plugin Models

Register external model packages via Python entry points:

```python
# In your package's pyproject.toml
[project.entry-points."lmms_eval.models"]
my_plugin = "my_package.models:get_manifests"
```

Or via the legacy environment variable (deprecated):
```bash
export LMMS_EVAL_PLUGINS=my_package
```

---

## 9. Debugging

### Quick Diagnostics

```bash
# Verbose logging
python -m lmms_eval --model qwen2_5_vl --tasks mme --limit 5 --verbosity DEBUG

# Save per-sample outputs for inspection
python -m lmms_eval --model qwen2_5_vl --tasks mme --limit 5 --log_samples --output_path ./debug_output
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: gen_kwargs['until']` | Wrong type for `until` in generation_kwargs | Must be `str` or `list[str]` |
| `NotImplementedError: loglikelihood` | Model doesn't support multiple-choice | Implement `loglikelihood()` or use `generate_until` tasks only |
| `AttributeError: '_max_length'` | Missing initialization | Set `self._max_length` in `__init__` |
| Visual is `None` or `[]` | Dataset sample has no image/video | Guard with `if visual is not None and len(visual) > 0` |
| API timeout/rate limit | API model hitting limits | Use `max_retries` and `retry_backoff_s` in model_args |
| `is_simple` mismatch | Chat model registered with `is_simple=True` | Set `is_simple = False` on chat model class |
| `CUDA out of memory` | Batch size or image resolution too large | Reduce `--batch_size`, set `max_pixels` in model_args |

### Logging

```python
from lmms_eval.utils import eval_logger

eval_logger.debug("Processing batch of {} samples", len(batch))
eval_logger.warning("Missing visual for doc_id={}", doc_id)
```

### Retry Patterns (API Models)

```bash
python -m lmms_eval --model openai --model_args pretrained=gpt-4o,max_retries=5,retry_backoff_s=2.0 --tasks mme
```

---

## 10. Model & Task Inventory

### Chat Models (14) - Recommended

| Model ID | Class | Backend |
|----------|-------|---------|
| `qwen2_5_vl` | Qwen2_5_VL | HuggingFace |
| `qwen3_vl` | Qwen3_VL | HuggingFace |
| `internvl_hf` | InternVLHf | HuggingFace |
| `llava_hf` | LlavaHf | HuggingFace |
| `llava_onevision1_5` | Llava_OneVision1_5 | HuggingFace |
| `huggingface` | Huggingface | HuggingFace (generic) |
| `openai` | OpenAICompatible | OpenAI API |
| `async_openai` | AsyncOpenAIChat | OpenAI API (async) |
| `vllm` | VLLM | vLLM |
| `vllm_generate` | VLLMGenerate | vLLM |
| `sglang` | Sglang | SGLang |
| `thyme` | Thyme | Custom |
| `longvila` | LongVila | Custom |
| `bagel_lmms_engine` | BagelLmmsEngine | lmms-engine |

### Simple Models (81) - Legacy

Major categories: Vision (30+), Video (15+), Audio (5+), Omni (4+), API (8+). Full list in `AVAILABLE_SIMPLE_MODELS` in `lmms_eval/models/__init__.py`.

### Task Categories

| Modality | Example Tasks | Count |
|----------|--------------|-------|
| Image | mme, mmmu, mmbench, ai2d, docvqa, textvqa, vqav2, pope | ~150 |
| Video | videomme, egoschema, mlvu, longvideobench, perceptiontest | ~40 |
| Audio | librispeech, clotho_aqa, covost2, fleurs, gigaspeech | ~25 |
| Reasoning | mathvista, mathverse, olympiadbench, gsm8k | ~15 |
| Language | arc, hellaswag, gpqa, ifeval | ~7 |

### Output Types

| Type | Description | Example Tasks |
|------|-------------|---------------|
| `generate_until` | Free-form text generation | mme, mmmu, mathvista |
| `loglikelihood` | Multiple-choice via perplexity | Some mmbench variants |
| `generate_until_multi_round` | Multi-turn conversation | mmsearch_end2end |

### Common model_args

| Argument | Description | Example |
|----------|-------------|---------|
| `pretrained` | Model checkpoint path | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `device_map` | GPU memory mapping | `auto` |
| `torch_dtype` | Data type | `bfloat16`, `float16` |
| `attn_implementation` | Attention type | `flash_attention_2`, `sdpa` |
| `max_pixels` | Max image pixels | `12845056` |
| `min_pixels` | Min image pixels | `3136` |
| `max_num_frames` | Max video frames | `32` |
| `trust_remote_code` | Allow remote code | `True` |
| `tensor_parallel_size` | vLLM tensor parallelism | `4` |

---

## 11. Constraints & Conventions

- **Package manager**: `uv` only, never `pip`
- **Formatting**: Black (line-length=240) + isort (profile=black). Run `pre-commit run --all-files`
- **No type suppression**: Never use `type: ignore`, `as any`, `@ts-ignore`
- **Commits**: Never mention co-authored-by or AI tools
- **Minimal changes**: Fix the specific issue, don't refactor unrelated code
- **Follow patterns**: Match the style of neighboring files exactly
- **PEP 8 naming**: snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- **Testing changes**: Always use `--limit 5` or `--limit 8` when testing model or task changes

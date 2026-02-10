# New Model Guide

To evaluate a model with `lmms_eval`, you implement a wrapper class that subclasses `lmms_eval.api.model.lmms`. This guide walks through the full process.

## Architecture Overview

```
    ╭──────────────╮                              ╭─────────────╮
    │  Model Dev   │                              │  Task Dev   │
    ╰──────┬───────╯                              ╰──────┬──────╯
           │                                             │
           ▼                                             ▼
 ┌─────────────────────┐                    ┌─────────────────────┐
 │                     │                    │                     │
 │  Implement lmms     │                    │  Create task YAML   │
 │  wrapper            │                    │  + utils.py         │
 │                     │                    │                     │
 │  Core methods:      │                    │  Preferred:         │
 │  · generate_until   │                    │  · doc_to_messages  │
 │  · loglikelihood    │                    │                     │
 │                     │                    │  Legacy:            │
 │  Register           │                    │  · doc_to_visual    │
 │  ModelManifest in   │                    │  · doc_to_text      │
 │  ModelRegistryV2    │                    │                     │
 └─────────┬───────────┘                    └──────────┬──────────┘
           │                                           │
           │           ╭ ─ ─ ─ ─ ─ ─ ─ ╮              │
           ╰──────────▶   Evaluator     ◀──────────────╯
                       │   contract     │
                        ╰ ─ ─ ─ ┬ ─ ─ ─╯
                                │
                                ▼
                  ┌───────────────────────┐
                  │                       │
                  │  Unified Instance     │
                  │  requests             │
                  │                       │
                  │  Model inference      │
                  │                       │
                  │  process_results      │
                  │                       │
                  │  metrics aggregation  │
                  │                       │
                  └───────────────────────┘
```

**Model Dev** implements the left side; **Task Dev** implements the right side. The evaluator runtime wires them together - your model never needs to know which task is calling it, and vice versa.

## Model Types

| Type | Location | Input method | Recommendation |
|------|----------|-------------|----------------|
| **Chat** | `models/chat/` | `doc_to_messages` - structured messages with roles and content types | **Use this** |
| **Simple** (legacy) | `models/simple/` | `doc_to_visual` + `doc_to_text` - plain text with `<image>` placeholders | Legacy only |

## Setup

```sh
git clone https://github.com/<YOUR-USERNAME>/lmms-eval.git
cd lmms-eval
git checkout -b <model-type>
pip install -e .

# Create your model file
touch lmms_eval/models/chat/<my_model>.py     # recommended
touch lmms_eval/models/simple/<my_model>.py   # legacy
```

Reference implementations: `lmms_eval/models/chat/qwen2_5_vl.py` (chat) and `lmms_eval/models/simple/instructblip.py` (simple).

## Core Methods

All models must subclass `lmms_eval.api.model.lmms` and implement two methods. Each receives a list of `Instance` objects (defined in [`lmms_eval.api.instance`](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/api/instance.py)) whose `.args` carry the request payload.

### `generate_until`

Open-ended generation. The model produces text given an input prompt + media.

**`Instance.args` for chat models** (5 elements):

| Element | Type | Description |
|---------|------|-------------|
| `doc_to_messages` | `Callable` | Function that converts a doc into structured `ChatMessages` |
| `gen_kwargs` | `dict` | Generation config: `max_new_tokens`, `temperature`, `until`, etc. |
| `doc_id` | `int` | Index into the dataset split |
| `task` | `str` | Task name (used to look up the dataset via `self.task_dict`) |
| `split` | `str` | Dataset split name |

**`Instance.args` for simple models** (6 elements):

| Element | Type | Description |
|---------|------|-------------|
| `contexts` | `str` | Formatted question text (may contain `<image>` tokens) |
| `gen_kwargs` | `dict` | Generation config |
| `doc_to_visual` | `Callable` | Function that returns a list of media (PIL images, video paths, etc.) |
| `doc_id` | `int` | Index into the dataset split |
| `task` | `str` | Task name |
| `split` | `str` | Dataset split name |

Returns `list[str]` - one generated string per request.

### `loglikelihood`

Scoring for multiple-choice tasks. The model computes the log-probability of a target continuation given a context.

**`Instance.args`** (6 elements):

| Element | Type | Description |
|---------|------|-------------|
| `contexts` | `str` | Formatted question text |
| `doc_to_target` | `Callable` | Function that extracts the answer continuation from the doc |
| `doc_to_visual` | `Callable` | Function that returns media |
| `doc_id` | `int` | Index into the dataset split |
| `task` | `str` | Task name |
| `split` | `str` | Dataset split name |

Returns `list[tuple[float, bool]]` - `(log_prob, is_greedy)` per request, where `is_greedy` is `True` if the target would be produced by greedy decoding.

## Registration

Register your model so `lmms_eval` can find it via `--model <name>`.

```python
from lmms_eval.api.registry import register_model

@register_model("my_model")
class MyModel(lmms):
    is_simple = False  # chat model (recommended)
    # is_simple = True  # simple model (legacy, default)
```

Then add the entry in `lmms_eval/models/__init__.py`:

```python
# Recommended (ModelRegistryV2 manifest)
from lmms_eval.models.registry_v2 import ModelManifest

MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="my_model",
        chat_class_path="lmms_eval.models.chat.my_model.MyModel",
    )
)

# Legacy (still supported)
AVAILABLE_CHAT_TEMPLATE_MODELS["my_model"] = "MyModel"
```

For external plugin packages, prefer Python entry-points (`lmms_eval.models`) over `LMMS_EVAL_PLUGINS`.

## Complete Example (Chat Model)

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages
import torch


@register_model("my_image_model")
class MyImageModel(lmms):
    is_simple = False

    def __init__(self, pretrained: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.device = device
        self.model = load_your_model(pretrained)
        self.processor = load_your_processor(pretrained)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            # Build structured messages from the doc
            doc = self.task_dict[task][split][doc_id]
            raw_messages = doc_to_messages(doc)
            messages = ChatMessages(messages=raw_messages)

            # Extract media and format prompt
            images, videos, audios = messages.extract_media()
            hf_messages = messages.to_hf_messages()
            text = self.processor.apply_chat_template(hf_messages)

            # Run inference
            inputs = self.processor(
                text=text, images=images, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                    temperature=gen_kwargs.get("temperature", 0.0),
                    do_sample=gen_kwargs.get("do_sample", False),
                )

            response = self.processor.decode(
                outputs[0], skip_special_tokens=True
            )
            results.append(response)
        return results

    def loglikelihood(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        results = []
        for request in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = (
                request.args
            )
            # Compute log-probability of the target continuation
            # given the context + visual inputs.
            # ...
        return results
```

For video and audio models the pattern is identical - the only difference is which media you extract from `messages.extract_media()`. See `lmms_eval/models/chat/qwen2_5_vl.py` for a production-quality reference.

## Key Notes

- Implement both `generate_until` and `loglikelihood` if your model supports generation and multiple-choice tasks
- Handle different modalities (image, video, audio) via the `ChatMessages` protocol
- Follow existing implementations in `lmms_eval/models/chat/` for patterns around batching, device management, and error handling

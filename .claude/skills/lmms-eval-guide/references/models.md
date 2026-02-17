<!-- lmms-eval v0.6 -->
# Adding a New Model

## Choose Model Type

| Type | Location | When to Use |
|------|----------|-------------|
| **Chat** (recommended) | `models/chat/` | New models, API models, anything with chat template |
| **Simple** (legacy) | `models/simple/` | Only if model has no chat template support |

Chat models receive structured `ChatMessages` via `doc_to_messages`. Simple models receive raw text + visual list.

## Chat Model Template

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

        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained, device_map=device_map, torch_dtype="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer

        self.batch_size_per_gpu = int(batch_size)
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._rank = 0
        self._world_size = 1

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for request in requests:
            # Chat models: 5 elements
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            doc = self.task_dict[task][split][doc_id]
            raw_messages = doc_to_messages(doc)
            messages = ChatMessages(messages=raw_messages)

            images, videos, audios = messages.extract_media()
            hf_messages = messages.to_hf_messages()
            text = self.processor.apply_chat_template(
                hf_messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=text,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
            ).to(self._device)

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", self._max_new_tokens),
                temperature=gen_kwargs.get("temperature", 0),
                do_sample=gen_kwargs.get("do_sample", False),
            )

            input_len = inputs["input_ids"].shape[1]
            response = self.processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
            results.append(response)
        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Required for multiple-choice tasks (loglikelihood output_type)
        raise NotImplementedError(
            "loglikelihood not implemented. Use generate_until tasks only, or implement this method."
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("generate_until_multi_round not implemented.")
```

## Register in `__init__.py`

Edit `lmms_eval/models/__init__.py`:

```python
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    # ... existing ...
    "my_model": "MyModel",  # Maps to lmms_eval.models.chat.my_model.MyModel
}

# Optional: backward-compatible aliases
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "my_model": ("my_model_v1", "old_name"),
}
```

## Test

```bash
python -m lmms_eval --model my_model --model_args pretrained=org/my-model --tasks mme --limit 5 --batch_size 1
python -m lmms_eval --model my_model --model_args pretrained=org/my-model --tasks mme --limit 5 --verbosity DEBUG
```

## Key Implementation Details

### Request Args Unpacking

```python
# Chat models: 5 elements
doc_to_messages, gen_kwargs, doc_id, task, split = request.args

# Simple models: 6 elements
contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
```

### `is_simple` Flag

- `False` for chat models (uses `doc_to_messages` -> `ChatMessages`)
- `True` (default) for simple models (uses `doc_to_visual` + `doc_to_text`)
- Registry validates this at load time

### `self.task_dict`

Populated by the framework before inference. Maps `task_name -> split -> doc_id -> document`.

### ChatMessages Protocol (`protocol.py`)

```python
messages = ChatMessages(messages=raw_messages)
images, videos, audios = messages.extract_media()    # Separate media
hf_messages = messages.to_hf_messages()              # For apply_chat_template()
oai_messages = messages.to_openai_messages()          # For OpenAI API (base64 images)
```

### Model Resolution

When `model_id` exists in both `AVAILABLE_CHAT_TEMPLATE_MODELS` and `AVAILABLE_SIMPLE_MODELS`, the registry creates one `ModelManifest` with both paths. Resolution prefers chat unless `force_simple=True`.

## Reference Implementations

| Pattern | File | Notes |
|---------|------|-------|
| HuggingFace local | `chat/qwen2_5_vl.py` | Standard transformers model |
| HuggingFace video | `chat/qwen3_vl.py` | Video-capable |
| API-based | `chat/openai.py` | OpenAI-compatible endpoints |
| vLLM backend | `chat/vllm.py` | High-throughput inference |
| SGLang backend | `chat/sglang.py` | Alternative inference engine |

## Common model_args

| Argument | Example | Description |
|----------|---------|-------------|
| `pretrained` | `Qwen/Qwen2.5-VL-7B-Instruct` | Checkpoint path |
| `device_map` | `auto` | GPU memory mapping |
| `torch_dtype` | `bfloat16` | Data type |
| `attn_implementation` | `flash_attention_2` | Attention backend |
| `max_pixels` | `12845056` | Max image pixels |
| `max_num_frames` | `32` | Max video frames |
| `tensor_parallel_size` | `4` | vLLM parallelism |

## Distributed Evaluation

```bash
# Multi-GPU with accelerate
accelerate launch --num_processes 4 -m lmms_eval --model qwen2_5_vl --tasks mme

# vLLM tensor parallelism
python -m lmms_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=4 --tasks mme
```

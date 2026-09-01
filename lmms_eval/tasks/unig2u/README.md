# UniG2U Benchmark

UniG2U (Unified Generation-to-Understanding) is a benchmark for evaluating multimodal models on their **unified understanding and generation capabilities**. It covers 11 sub-tasks spanning chart understanding, geometry reasoning, physics analysis, visual-spatial planning, and more. The goal is to measure whether models benefit from a two-stage pipeline: first generating auxiliary visualizations, then answering questions with them.

## Task Overview

UniG2U provides two evaluation modes:

| Mode | Task Name | Description |
|------|-----------|-------------|
| **Standard** | `unig2u` | Direct image understanding (single-stage) |
| **GtA (Generation-to-Answer)** | `unig2u_GtA` | Generate auxiliary image first, then answer (two-stage) |

### Sub-tasks

| Sub-task | Domain | Samples |
|----------|--------|---------|
| ChartQA100 | Chart understanding | 100 |
| Geometry3K | Plane geometry | -- |
| AuxSolidMath | Solid geometry (auxiliary lines) | -- |
| BabyVision | Fine-grained visual discrimination + tracking | 246 |
| IllusionBench | Illusion art recognition | -- |
| MMSI-Bench | Multimodal spatial intelligence (5 sub-tasks) | 500 |
| PhyX | Physics (optics + mechanics) | 200 |
| RealUnify | Cognitive psychology (tracking/reconstruction/focusing) | 300 |
| Uni-MMMU | Interleaved generation (jigsaw/maze/sliding) | 254 |
| VSP | Visual-spatial planning (collision/navigation) | 100 |
| VisualPuzzles | Visual reasoning (5 reasoning types) | 500 |

## Quick Start

```bash
# Standard understanding - all 11 sub-tasks
python -m lmms_eval \
    --model ovis_u1 \
    --model_args pretrained=AIDC-AI/Ovis-U1-3B \
    --tasks unig2u \
    --batch_size 1

# GtA (two-stage) - all 11 sub-tasks
python -m lmms_eval \
    --model ovis_u1 \
    --model_args pretrained=AIDC-AI/Ovis-U1-3B \
    --tasks unig2u_GtA \
    --batch_size 1

# Single sub-task
python -m lmms_eval \
    --model bagel_unig2u \
    --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=understanding \
    --tasks unig2u_chartqa100 \
    --batch_size 1

# Quick smoke test (limit samples)
python -m lmms_eval \
    --model ovis_u1 \
    --model_args pretrained=AIDC-AI/Ovis-U1-3B \
    --tasks unig2u_chartqa100 \
    --batch_size 1 \
    --limit 2
```

## GtA Two-Stage Pipeline

In GtA mode, the model performs two-step inference:

```
Stage 1: Original image + generation prompt -> Generate auxiliary visualization
Stage 2: Original image + auxiliary image + question -> Answer
```

GtA tasks use a dedicated output type `generate_visual_cot`. The evaluator calls `model.generate_visual_cot()` instead of `model.generate_until()`. Models that do not implement this method will get a clear error:

```
NotImplementedError: MyModel does not support Visual CoT (GtA).
To run visual_cot tasks, the model must implement generate_visual_cot().
```

```yaml
# Standard YAML (single-stage)
output_type: generate_until
generation_kwargs:
  max_new_tokens: 16
  temperature: 0

# GtA YAML (two-stage)
output_type: generate_visual_cot    # <- dedicated output type
generation_kwargs:
  visual_cot: true
  max_new_tokens: 16
  temperature: 0
```

## Supported Models

| Model | model name | Standard | GtA | Notes |
|-------|-----------|:---:|:---:|-------|
| Ovis-U1 | `ovis_u1` | Y | Y | Conditional generation + understanding |
| Bagel | `bagel_unig2u` | Y | Y | mode=understanding/generation |
| ILLUME+ | `illume_plus` | Y | Y | enable_visual_cot + decoder required for GtA |
| MMaDa | `mmada` | Y | -- | Understanding only, no GtA |
| Qwen-Image-Edit | `qwen_image_edit` | Y | Y | Edit model for Stage 1 |

## Adding a New Model

### 1. Standard Understanding

Inherit from `lmms` and implement `generate_until()`:

```python
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("my_model")
class MyModel(lmms):
    def generate_until(self, requests):
        """
        Input: List[Instance], each instance.args contains:
            (context, gen_kwargs, doc_to_visual, doc_id, task, split)
        Output: List[str], generated text answers
        """
        res = []
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            images = doc_to_visual(doc)  # List[PIL.Image]
            answer = self._inference(context, images, gen_kwargs)
            res.append(answer)
        return res
```

Register in `lmms_eval/models/__init__.py`:

```python
AVAILABLE_SIMPLE_MODELS = {
    ...
    "my_model": "MyModel",
}
```

Then run: `--tasks unig2u --model my_model`

### 2. GtA Support (Optional)

If your model supports image generation, implement `generate_visual_cot()`:

```python
def generate_visual_cot(self, requests):
    """Required for GtA tasks (output_type: generate_visual_cot)."""
    return self.generate_until(requests)
```

Inside `generate_until`, detect GtA mode via `[GEN_PROMPT]` tags in the prompt or `visual_cot: true` in gen_kwargs, then run the two-stage pipeline:

```python
def generate_until(self, requests):
    res = []
    for request in requests:
        context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

        if gen_kwargs.pop("visual_cot", False) or "[GEN_PROMPT]" in context:
            answer = self._visual_cot_pipeline(context, doc_to_visual, doc_id, task, split)
        else:
            answer = self._standard_inference(context, doc_to_visual, doc_id, task, split)

        res.append(answer)
    return res
```

The two-stage pipeline:

```python
def _visual_cot_pipeline(self, context, doc_to_visual, doc_id, task, split):
    import re
    doc = self.task_dict[task][split][doc_id]
    original_images = doc_to_visual(doc)

    # Parse generation prompt and question from context
    gen_match = re.search(r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", context, re.DOTALL)
    q_match   = re.search(r"\[QUESTION\](.*?)\[/QUESTION\]", context, re.DOTALL)

    gen_prompt = gen_match.group(1).strip() if gen_match else context
    question   = q_match.group(1).strip() if q_match else context

    # Stage 1: Generate auxiliary image
    auxiliary_image = self._generate_image(gen_prompt, original_images)

    # Stage 2: Answer with auxiliary image
    all_images = original_images + [auxiliary_image]
    answer = self._understand(question, all_images)
    return answer
```

### 3. Verify

```bash
# Standard understanding
python -m lmms_eval --model my_model \
    --model_args pretrained=... \
    --tasks unig2u_chartqa100 --batch_size 1 --limit 2

# GtA (requires generate_visual_cot)
python -m lmms_eval --model my_model \
    --model_args pretrained=... \
    --tasks unig2u_chartqa100_visual_cot --batch_size 1 --limit 2
```

## File Structure

```
tasks/unig2u/
├── unig2u.yaml                  # Standard group (11 sub-tasks)
├── unig2u_GtA.yaml              # GtA group (11 visual_cot sub-tasks)
├── chartqa100.yaml              # Sub-task YAML (standard, output_type: generate_until)
├── chartqa100_visual_cot.yaml   # Sub-task YAML (GtA, output_type: generate_visual_cot)
├── mmsi_*.yaml                  # MMSI-Bench series
├── ...                          # Other sub-task YAMLs
├── utils.py                     # Merged processing functions for all sub-tasks
├── arshia_utils.py              # IllusionBench processing functions
└── README.md                    # This file
```

## Dataset

All data hosted on HuggingFace: `hf://datasets/kkv233/unig2u_dataset/`

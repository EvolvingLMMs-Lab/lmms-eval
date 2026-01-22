# BabyVision Benchmark

> *Can MLLMs See Like a 3-Year-Old?*

BabyVision is a benchmark for evaluating multimodal LLMs on core visual reasoning tasks that even 3-year-old children can solve. It tests fundamental visual abilities independent of linguistic knowledge.

## Overview

- **388 visual reasoning questions** across 4 categories and 22 subtypes
- **LLM-as-Judge evaluation** for semantic answer matching
- **Paper**: [BabyVision: Visual Reasoning Beyond Language](https://arxiv.org/abs/2601.06521)
- **Dataset**: [UnipatAI/BabyVision](https://huggingface.co/datasets/UnipatAI/BabyVision)

## Task Categories

| Category | Subtypes | Examples |
|----------|----------|----------|
| Fine-grained Discrimination | Find the different, Find the same, Find the shadow, Count patterns, etc. | Pattern matching, shadow finding |
| Visual Tracking | Maze, Connect the lines, Metro map, Recognize numbers/letters | Line tracing, path finding |
| Spatial Perception | 3D Views, 3D Cube Unfold, Paper Folding, Count 3D blocks | 3D reasoning |
| Visual Pattern Recognition | Logic Patterns, Rotation, Mirroring, Overlay | Pattern completion |

## Usage

```bash
# Set up API for LLM-as-Judge (required)
export BABYVISION_API_KEY="your-api-key"
export BABYVISION_BASE_URL="https://api.openai.com/v1"  # optional
export BABYVISION_MODEL_NAME="gpt-4o"  # optional, default: gpt-4o

# Run evaluation
lmms-eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks babyvision \
    --batch_size 1 \
    --output_path ./logs/babyvision
```

## Metrics

| Metric | Description |
|--------|-------------|
| `babyvision_overall_accuracy` | Overall accuracy across all 388 questions |

Per-subtype breakdown is logged during evaluation for diagnostic insights.

## Answer Format

Models are prompted to provide answers in `\boxed{Answer}` format:

- **Multiple choice**: `\boxed{A}`, `\boxed{B}`, etc.
- **Fill-in-blank**: `\boxed{(4,7)}`, `\boxed{5}`, etc.

The evaluation supports models with extended reasoning (e.g., `<think>...</think>` blocks) by extracting the final answer after reasoning.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BABYVISION_API_KEY` | Yes | - | API key for LLM judge |
| `BABYVISION_BASE_URL` | No | `https://api.openai.com/v1` | API base URL |
| `BABYVISION_MODEL_NAME` | No | `gpt-4o` | Judge model name |

## Related

- **BabyVision-Gen**: Image generation evaluation task (separate PR)

## Citation

```bibtex
@misc{chen2026babyvisionvisualreasoninglanguage,
      title={BabyVision: Visual Reasoning Beyond Language}, 
      author={Liang Chen and Weichu Xie and ...},
      year={2026},
      eprint={2601.06521},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.06521}, 
}
```

# Uni-MMMU

Uni-MMMU evaluates unified multimodal models on bidirectional synergy between generation and understanding.

- Paper: https://arxiv.org/abs/2510.13759
- Dataset: https://huggingface.co/datasets/lmms-lab-eval/UniMMMU

## Tasks

- **jigsaw**: Complete a 2x2 image puzzle by selecting the correct patch
- **maze**: Find path through a maze
- **sliding**: Solve sliding puzzle
- **geometry**: Solve geometry problems with visual diagrams

## Usage

```bash
python -m lmms_eval --model <model> --tasks uni_mmmu --limit 10
```

## Metrics

- `exact_match`: Accuracy of exact match
- `frame_accuracy`: For maze/sliding, step-by-step accuracy

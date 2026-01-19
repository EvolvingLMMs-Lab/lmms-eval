# VLMs Are Blind

## Overview

VLMs Are Blind is a benchmark designed to test the visual reasoning capabilities of Vision-Language Models (VLMs) through path-counting tasks in subway connection diagrams. The benchmark reveals fundamental limitations in VLMs' ability to process visual information, showing that many models struggle with basic visual tasks that are trivial for humans.

## Paper Information

- **Paper Title**: VLMs Are Blind
- **Paper**: https://arxiv.org/abs/2407.06581
- **GitHub**: https://github.com/xai-org/vlmsareblind
- **Dataset**: https://huggingface.co/datasets/XAI/vlmsareblind

## Dataset Details

The benchmark consists of path-counting tasks where models must count the number of paths between two stations in subway-style connection diagrams. Each instance contains:
- A subway map diagram image
- A question asking for the number of paths between two specified stations
- The correct answer in the format {N}

## Task Configuration

### Main Task
- **Task Name**: `vlmsareblind`
- **Split**: `valid`
- **Output Type**: `generate_until`
- **Metric**: Exact match accuracy

### Lite Version
- **Task Name**: `vlmsareblind_lite`
- A subset version for faster evaluation

## Evaluation

The benchmark uses exact match accuracy as the primary metric. Models must output their answer in the format `{N}` where N is the number of paths.

### Metrics
- **exact_match**: Binary score for each instance (1 if prediction matches ground truth, 0 otherwise)
- **Aggregation**: Mean across all instances

## Running the Benchmark

```bash
# Run the full benchmark
lmms-eval --model <model_name> --tasks vlmsareblind --batch_size 1

# Run the lite version
lmms-eval --model <model_name> --tasks vlmsareblind_lite --batch_size 1
```

## Implementation Details

### Generation Configuration
- **max_new_tokens**: 32
- **temperature**: 0
- **top_p**: 1.0
- **num_beams**: 1
- **do_sample**: false

### Answer Extraction
The evaluation expects answers in the format `{N}`. The answer extraction logic:
1. First looks for numbers within curly brackets: `{3}`
2. If not found, looks for standalone numbers and adds brackets
3. Falls back to the raw response if no pattern matches

### Prompt Format
```
[Image of subway diagram]
[Question about counting paths]
Answer with a number in curly brackets, e.g., {3}.
```

## File Structure
```
vlmsareblind/
├── README.md           # This file
├── utils.py           # Evaluation utilities
├── vlmsareblind.yaml  # Main task configuration
└── vlmsareblind_lite.yaml  # Lite version configuration
```

## Citation

```bibtex
@article{vlmsareblind2024,
  title={VLMs Are Blind},
  author={XAI Team},
  journal={arXiv preprint arXiv:2407.06581},
  year={2024}
}
```

## Notes

- The benchmark is designed to test fundamental visual reasoning capabilities
- Results often reveal significant gaps between VLM performance and human abilities
- The simple counting task format makes it easy to verify model outputs
- Temperature is set to 0 for deterministic outputs
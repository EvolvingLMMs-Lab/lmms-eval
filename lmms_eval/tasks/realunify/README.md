# RealUnify: Unified Multimodal Model Evaluation

## Overview

RealUnify is a benchmark designed to evaluate bidirectional capability synergy in unified multimodal models.

- **Paper**: [RealUnify: Do Unified Models Truly Benefit from Unification?](https://arxiv.org/abs/2509.24897)
- **Dataset**: [lmms-lab-eval/RealUnify](https://huggingface.co/datasets/lmms-lab-eval/RealUnify)
- **GitHub**: [FrankYang-17/RealUnify](https://github.com/FrankYang-17/RealUnify)

## Dataset

The dataset is available at [lmms-lab-eval/RealUnify](https://huggingface.co/datasets/lmms-lab-eval/RealUnify) with configs for each task type:
- `mental_tracking` (100 samples)
- `mental_reconstruction` (100 samples)
- `attentional_focusing` (100 samples)
- `cognitive_navigation` (100 samples)

## Benchmark Structure

RealUnify comprises **1,000 human-annotated instances** spanning:
- 10 categories
- 32 subtasks

### Evaluation Axes

1. **Understanding Enhances Generation (UEG)**: Requires reasoning to guide image generation
2. **Generation Enhances Understanding (GEU)**: Requires mental simulation to solve reasoning tasks

### Implemented Tasks (GEU)

- `realunify_mental_tracking`: Mental tracking tasks requiring visual transformation reasoning
- `realunify_mental_reconstruction`: Tasks where shuffled images need mental reconstruction
- `realunify_attentional_focusing`: Tasks requiring attention to specific image regions

## Usage

```bash
python -m lmms_eval --tasks realunify --model <model_name> --model_args <args>

python -m lmms_eval --tasks realunify_mental_tracking --model <model_name>
```

## Citation

```bibtex
@misc{shi2025realunifyunifiedmodelstruly,
      title={RealUnify: Do Unified Models Truly Benefit from Unification?},
      author={Yang Shi and Yuhao Dong and others},
      year={2025},
      eprint={2509.24897},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.24897},
}
```

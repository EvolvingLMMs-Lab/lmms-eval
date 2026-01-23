# RealUnify: Unified Multimodal Model Evaluation

## Overview

RealUnify is a benchmark designed to evaluate bidirectional capability synergy in unified multimodal models.

- **Paper**: [RealUnify: Do Unified Models Truly Benefit from Unification?](https://arxiv.org/abs/2509.24897)
- **Dataset**: [DogNeverSleep/RealUnify](https://huggingface.co/datasets/DogNeverSleep/RealUnify)
- **GitHub**: [FrankYang-17/RealUnify](https://github.com/FrankYang-17/RealUnify)

## Data Preparation

The RealUnify benchmark requires parquet files with annotations. You need to prepare the data in one of these ways:

1. **Request from authors**: Contact `frankyang1517@gmail.com` for the full dataset with annotations
2. **Convert from JSON**: Download `GEU_direct.json` from the official repo and convert to parquet

Place the parquet files in your data directory:
```
${LMMS_EVAL_DATA_DIR}/realunify/
├── realunify_mental_tracking.parquet
├── realunify_mental_reconstruction.parquet
└── realunify_attentional_focusing.parquet
```

Set the environment variable:
```bash
export LMMS_EVAL_DATA_DIR=/path/to/your/data
```

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

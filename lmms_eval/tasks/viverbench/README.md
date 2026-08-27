# ViVerBench

ViVerBench evaluates whether multimodal models can verify if generated visual outputs satisfy prompt-level constraints.

- Paper: [Generative Universal Verifier as Multimodal Meta-Reasoner](https://huggingface.co/papers/2510.13804)
- Dataset: [comin/ViVerBench](https://huggingface.co/datasets/comin/ViVerBench)

## Overview

- 3,594 examples across 16 task categories
- Binary verification target (`true` / `false`)
- Inputs can contain multiple images (1, 2, or 8)

## Usage

```bash
python -m lmms_eval \
  --model <model_name> \
  --tasks viverbench \
  --batch_size 1 \
  --limit 8
```

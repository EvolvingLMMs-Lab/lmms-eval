# WISE

WISE is a knowledge-intensive text-to-image benchmark that evaluates whether models can use commonsense, cultural, scientific, spatial, and temporal knowledge to generate correct images.

- Paper: https://arxiv.org/abs/2503.07265
- Dataset: https://huggingface.co/datasets/Yuwei-Niu/WISE

## Overview

**Dataset:** 1000 prompts across 6 categories (Culture, Time, Space, Biology, Physics, Chemistry)

**Evaluation:** GPT-4o judges each generated image on three dimensions:
- **Consistency** (0-2): How well the image matches the prompt
- **Realism** (0-2): Visual quality and photorealism
- **Aesthetic Quality** (0-2): Artistic appeal and composition

**WiScore Formula:** `(0.7 × consistency + 0.2 × realism + 0.1 × aesthetic) / 2`

**Final Score:** Weighted average across categories (Culture: 0.4, Time: 0.167, Space: 0.133, Biology/Physics/Chemistry: 0.1 each)

## Environment Variables

```bash
export WISE_API_KEY="your-api-key"                    # Judge API key
export WISE_BASE_URL="https://api.openai.com/v1"     # Judge API endpoint
export WISE_MODEL_NAME="gpt-4o-2024-05-13"                       # Judge model name
export WISE_RAW_OUTPUT_DIR="/path/to/model/output"   # Where model saves generated images
```

## Usage

### Full Evaluation with bagel

```bash
cd /pfs/weiyang/Show/lmms-eval

export WISE_API_KEY="sk-..."
export WISE_BASE_URL="https://api.bltcy.ai/v1"
export WISE_MODEL_NAME="gpt-4o"
export WISE_RAW_OUTPUT_DIR="/pfs/weiyang/Show/lmms-eval/outputs/WISE_raw/bagel_umm"

python -m lmms_eval \
  --model bagel_umm \
  --model_args pretrained=/pfs/weiyang/WISE_re/CKPT/ByteDance-Seed/BAGEL-7B-MoT,mode=generate,output_dir=/pfs/weiyang/Show/lmms-eval/outputs/WISE_raw/bagel_umm \
  --tasks WISE \
  --batch_size 1 \
  --log_samples \
  --output_path /pfs/weiyang/Show/lmms-eval/outputs/WISE_eval/bagel_umm
```

## Metrics

- `WISE_culture_score`: Culture category score (prompt_id 1-400)
- `WISE_time_score`: Time category score (prompt_id 401-567)
- `WISE_space_score`: Space category score (prompt_id 568-700)
- `WISE_biology_score`: Biology category score (prompt_id 701-800)
- `WISE_physics_score`: Physics category score (prompt_id 801-900)
- `WISE_chemistry_score`: Chemistry category score (prompt_id 901-1000)
- `WISE_overall_wiscore`: Weighted overall score (main metric)

All scores are in the range [0.0, 1.0].

Do not use VQA-only models or image-editing models for this task.

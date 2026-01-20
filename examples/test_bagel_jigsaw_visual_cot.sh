#!/bin/bash
# Test Bagel on Uni-MMMU Jigsaw Visual CoT (first 5 samples)

python -m lmms_eval \
  --model bagel \
  --model_args pretrained=./BAGEL-7B-MoT,mode=generation \
  --tasks uni_mmmu_jigsaw100_visual_cot \
  --batch_size 1 \
  --output_path ./logs/bagel_jigsaw \
    --log_samples \
    --limit 1 \
    --verbosity INFO

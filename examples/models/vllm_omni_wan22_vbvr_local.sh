#!/usr/bin/env bash

./.venv/bin/python -m lmms_eval \
  --model vllm_omni \
  --model_args model=Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --tasks vbvr \
  --batch_size 1 \
  --log_samples \
  --output_path logs

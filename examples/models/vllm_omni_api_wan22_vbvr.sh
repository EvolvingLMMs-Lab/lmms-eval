#!/usr/bin/env bash

./.venv/bin/python -m lmms_eval \
  --model vllm_omni_api \
  --tasks vbvr \
  --batch_size 1 \
  --log_samples \
  --output_path logs

#!/usr/bin/env bash

./.venv-sglang/bin/python -m lmms_eval \
  --model sglang \
  --model_args model=Wan-AI/Wan2.2-I2V-A14B-Diffusers,runtime=diffusion,num_gpus=4,enable_cfg_parallel=true,ulysses_degree=2,text_encoder_cpu_offload=true,pin_cpu_memory=true \
  --tasks vbvr \
  --batch_size 1 \
  --log_samples \
  --output_path logs

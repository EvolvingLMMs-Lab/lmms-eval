#!/usr/bin/env bash
set -euo pipefail
# Run and reproduce Thyme results!
# MME-RealWorld-Lite as an example
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils autopep8 timeout_decorator

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model thyme \
    --model_args=pretrained=Kwai-Keye/Thyme-RL,min_pixels=3136,max_pixels=12845056,attn_implementation=sdpa,interleave_visuals=False \
    --tasks mmerealworld_lite \
    --batch_size 1
#!/bin/bash

# GPU ID (default: 0)
GPU_ID=${1:-0}

# Set environment variables
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MASTER_PORT=2950${GPU_ID}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

echo "Running UniWorld on Uni-MMMU Jigsaw Visual CoT using GPU: ${GPU_ID}"
echo "MASTER_PORT: ${MASTER_PORT}"

# UniWorld model path
MODEL_PATH=${MODEL_PATH:-"LanguageBind/UniWorld-V1"}

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} -m lmms_eval \
  --model uniworld \
  --model_args pretrained=${MODEL_PATH},image_output_dir=./logs/uniworld_jigsaw/images \
  --tasks uni_mmmu_jigsaw100_visual_cot \
  --batch_size 1 \
  --output_path ./logs/uniworld_jigsaw \
  --log_samples \
  --verbosity INFO

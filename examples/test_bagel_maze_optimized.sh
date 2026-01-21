#!/bin/bash
# Memory-optimized Bagel on Uni-MMMU Maze Visual CoT

GPU_ID=${1:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MASTER_PORT=2950${GPU_ID}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

# Memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Running on GPU: ${GPU_ID} (Memory Optimized)"
echo "MASTER_PORT: ${MASTER_PORT}"

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} -m lmms_eval \
  --model bagel \
  --model_args pretrained=./BAGEL-7B-MoT,mode=generation,output_image_dir=/mnt/data/bagel_maze/images,device_map=auto,torch_dtype=bfloat16,low_cpu_mem_usage=true \
  --tasks uni_mmmu_maze100_visual_cot \
  --batch_size 1 \
  --output_path /mnt/data/bagel_maze \
  --log_samples \
  --verbosity INFO

# Usage:
# bash examples/test_bagel_maze_optimized.sh 0

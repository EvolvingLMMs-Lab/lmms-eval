#!/bin/bash
# Multi-GPU test for Bagel on Uni-MMMU Maze Visual CoT

# Number of GPUs to use
NUM_GPUS=${1:-2}  # Default: 2 GPUs

# Set environment variables
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

echo "Running on ${NUM_GPUS} GPUs"
echo "MASTER_PORT: ${MASTER_PORT}"

# Use accelerate for multi-GPU inference
accelerate launch \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MASTER_PORT} \
  -m lmms_eval \
  --model bagel \
  --model_args pretrained=./BAGEL-7B-MoT,mode=generation,output_image_dir=/mnt/data/bagel_maze/images \
  --tasks uni_mmmu_maze100_visual_cot \
  --batch_size 1 \
  --output_path /mnt/data/bagel_maze \
  --log_samples \
  --verbosity INFO

# Usage:
# bash examples/test_bagel_maze_multiGPU.sh 2   # Use 2 GPUs
# bash examples/test_bagel_maze_multiGPU.sh 4   # Use 4 GPUs

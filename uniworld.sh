#!/bin/bash
# UniWorld Model - ChartQA100 Evaluation Script
#
# This script evaluates UniWorld-V1 on ChartQA100 benchmark
# UniWorld is a unified understanding-generation model
#
# Prerequisites:
#   1. Clone UniWorld-V1 repository at lmms-eval root:
#      cd /path/to/lmms-eval
#      git clone https://github.com/LanguageBind/UniWorld.git
#
# Usage:
#   bash uniworld.sh [GPU_IDS] [MODEL_PATH] [OUTPUT_PATH]
#
# Examples:
#   bash uniworld.sh "0"                    # Single GPU
#   bash uniworld.sh "0,1"                  # Multi-GPU (2 GPUs)
#   bash uniworld.sh "0,1,2,3"              # Multi-GPU (4 GPUs)
#   bash uniworld.sh "0" "./UniWorld-V1"    # Custom model path
#   bash uniworld.sh "0,1" "LanguageBind/UniWorld-V1" "./my_output"  # All custom

# ============ Configuration ============
# GPU IDs to use (comma-separated for multi-GPU)
GPU_IDS=${1:-"0"}

# Model path (can be HuggingFace model ID or local path)
MODEL_PATH=${2:-"LanguageBind/UniWorld-V1"}

# Output directory
OUTPUT_PATH=${3:-"./logs/uniworld_chartqa100"}

# Task to evaluate
TASK="chartqa100"

# Batch size (keep at 1 for generation tasks)
BATCH_SIZE=1

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=29601
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

# ============ Print Configuration ============
echo "======================================"
echo "UniWorld - ChartQA100 Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task:          ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Master Port:   ${MASTER_PORT}"
echo "======================================"
echo ""

# ============ Run Evaluation ============
# Use accelerate launch for better multi-GPU support
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model uniworld \
  --model_args pretrained=${MODEL_PATH} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  --verbosity INFO

# Alternative: Use torchrun (uncomment if needed)
# torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} -m lmms_eval \
#   --model uniworld \
#   --model_args pretrained=${MODEL_PATH} \
#   --tasks ${TASK} \
#   --batch_size ${BATCH_SIZE} \
#   --output_path ${OUTPUT_PATH} \
#   --log_samples \
#   --verbosity INFO

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "======================================"

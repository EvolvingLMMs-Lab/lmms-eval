#!/bin/bash

# Bagel Model Evaluation Script for GEdit-Bench
#
# This script demonstrates how to run lmms-eval with the Bagel multimodal model
# for image editing tasks using GEdit-Bench.
#
# Prerequisites:
#   1. Clone Bagel repository at lmms-eval root:
#      cd /path/to/lmms-eval
#      git clone https://github.com/ByteDance-Seed/Bagel.git
#
#   2. Model weights can be anywhere (specify via MODEL_PATH below)
#      Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
#
# Usage:
#   # Use local Qwen2.5-VL for evaluation:
#   bash examples/models/bagel.sh qwen25vl
#
#   # Use vLLM remote Qwen for evaluation:
#   bash examples/models/bagel.sh vllm_qwen
#
#   # Use GPT-4o for evaluation:
#   bash examples/models/bagel.sh gpt4o

# Activate conda environment (uncomment and modify if needed)
# source miniconda3/etc/profile.d/conda.sh
# conda activate lmms-eval

# ============================================
# Configuration
# ============================================

MODEL_PATH=/your/path/to/models/BAGEL-7B-MoT
TASK=gedit_bench

# GEdit-Bench environment variables
export GEDIT_BENCH_MODEL_NAME="bagel"
export GEDIT_BENCH_OUTPUT_DIR="./logs/bagel_persistent_folder/bagel_generated_images"
export GEDIT_BENCH_VIE_KEY_PATH="./lmms_eval/tasks/gedit_bench/secret.env"

# ============================================
# Evaluation Backend Selection
# ============================================

# Get backend from command line argument, default to "qwen25vl"
EVAL_BACKBONE=${1:-vllm_qwen25vl}

if [ "$EVAL_BACKBONE" == "vllm_qwen" ] || [ "$EVAL_BACKBONE" == "vllm_qwen25vl" ] || [ "$EVAL_BACKBONE" == "vllm_qwen3vl" ]; then
    echo "Using vLLM Qwen for VIEScore evaluation..."
    export GEDIT_BENCH_VIE_BACKBONE="$EVAL_BACKBONE"
    # vLLM API settings - modify these for your setup
    export VLLM_API_BASE="http://host:8000/v1"
    # export VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
    export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
    export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
    echo "  VLLM_API_BASE: $VLLM_API_BASE"
    echo "  VLLM_MODEL_NAME: $VLLM_MODEL_NAME"
elif [ "$EVAL_BACKBONE" == "gpt4o" ]; then
    echo "Using GPT-4o for VIEScore evaluation..."
    export GEDIT_BENCH_VIE_BACKBONE="gpt4o"
    # Set your OpenAI API key
    # export OPENAI_API_KEY="your-api-key-here"
else
    echo "Using local Qwen2.5-VL for VIEScore evaluation..."
    export GEDIT_BENCH_VIE_BACKBONE="qwen25vl"
fi

# ============================================
# Run Evaluation
# ============================================

echo "============================================"
echo "Starting GEdit-Bench evaluation..."
echo "============================================"
echo "  Model: Bagel"
echo "  Model Path: $MODEL_PATH"
echo "  Evaluation Backend: $GEDIT_BENCH_VIE_BACKBONE"
echo "  Output Directory: $GEDIT_BENCH_OUTPUT_DIR"
echo "============================================"
echo ""

# 图像编辑任务 (GEdit-Bench)
# task_mode=edit: 输入图像 + 编辑指令 -> 编辑后的图像
accelerate launch -m lmms_eval \
    --model bagel \
    --model_args pretrained=${MODEL_PATH},task_mode=edit \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "============================================"

# 如果是文本生图任务，使用 task_mode=generate:
# accelerate launch -m lmms_eval \
#     --model bagel \
#     --model_args pretrained=${MODEL_PATH},task_mode=generate \
#     --tasks ueval \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./logs/

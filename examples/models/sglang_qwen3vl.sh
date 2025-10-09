#!/bin/bash

# Qwen3-VL Evaluation Script with SGLang Backend
# This script demonstrates how to evaluate Qwen3-VL models using SGLang for accelerated inference
#
# Requirements:
# - sglang>=0.4.6
# - qwen-vl-utils
# - CUDA-enabled GPU(s)
#
# Installation:
# uv add "sglang[all]" qwen-vl-utils
# OR
# pip install "sglang[all]>=0.4.6" qwen-vl-utils

# ============================================================================
# Configuration
# ============================================================================

# Model Configuration
# Available Qwen3-VL models:
# - Qwen/Qwen3-VL-30B-A3B-Instruct
# - Qwen/Qwen3-VL-30B-A3B-Thinking
# - Qwen/Qwen3-VL-235B-A22B-Instruct
# - Qwen/Qwen3-VL-235B-A22B-Thinking
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"

# Parallelization Settings
# Adjust based on your GPU configuration
TENSOR_PARALLEL_SIZE=4  # Number of GPUs for tensor parallelism (tp_size in SGLang)

# Memory and Performance Settings
GPU_MEMORY_UTILIZATION=0.85  # mem_fraction_static in SGLang (0.0 - 1.0)
BATCH_SIZE=64                # Batch size for evaluation

# SGLang Specific Settings
MAX_PIXELS=1605632           # Maximum pixels for image processing
MIN_PIXELS=784               # Minimum pixels (28x28)
MAX_FRAME_NUM=32            # Maximum number of video frames
THREADS=16                  # Number of threads for decoding visuals

# Task Configuration
# Common tasks: mmmu_val, mme, mathvista, ai2d, etc.
TASKS="mmmu_val,mme"

# Output Configuration
OUTPUT_PATH="./logs/qwen3vl_sglang"
LOG_SAMPLES=true
LOG_SUFFIX="qwen3vl_sglang"

# Evaluation Limits (optional)
# LIMIT=100  # Uncomment to limit number of samples (for testing)

# ============================================================================
# Environment Configuration
# ============================================================================
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ============================================================================
# Run Evaluation
# ============================================================================

echo "=========================================="
echo "Qwen3-VL Evaluation with SGLang"
echo "=========================================="
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Pixels: $MAX_PIXELS"
echo "Output Path: $OUTPUT_PATH"
echo "=========================================="

# Build the command
CMD="uv run python -m lmms_eval \
    --model sglang_runtime \
    --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_frame_num=${MAX_FRAME_NUM},threads=${THREADS} \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}"

# Add optional arguments
if [ "$LOG_SAMPLES" = true ]; then
    CMD="$CMD --log_samples --log_samples_suffix ${LOG_SUFFIX}"
fi

if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit ${LIMIT}"
fi

# Execute
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================="

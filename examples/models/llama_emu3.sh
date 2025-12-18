#!/bin/bash

# Llama-Emu3 Benchmark Script
# This script runs lmms-eval benchmarks for the Llama-Emu3 model with Emu3 image tokenization
#
# Usage:
#   ./llama_emu3.sh [PRETRAINED] [TOKENIZER_PATH] [TASKS] [BATCH_SIZE] [NUM_PROCESSES]
#
# Examples:
#   ./llama_emu3.sh                                                    # Use all defaults
#   ./llama_emu3.sh /path/to/model                                    # Custom model path
#   ./llama_emu3.sh /path/to/model /path/to/tokenizer                # Custom model and tokenizer
#   ./llama_emu3.sh /path/to/model /path/to/tokenizer mme,ai2d 4 8  # Full customization

# ========== Configuration Parameters ==========

# Model paths (required - adjust these to your setup)
PRETRAINED="${1:-/iopsstor/scratch/cscs/nirmiger/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-2n-8192sl-120gbsz-0.5-0.5/HF}"
TOKENIZER_PATH="${2:-/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer}"

# Task configuration
TASKS="${3:-ai2d,chartqa,docvqa_val,mmmu_pro}"

# Batch and process configuration
BATCH_SIZE="${4:-1}"
NUM_PROCESSES="${5:-8}"
MAIN_PROCESS_PORT="${6:-12399}"

# Model configuration (advanced - usually don't need to change)
MAX_LENGTH="${MAX_LENGTH:-}"                     # Empty = auto-detect from model config (8192 for Llama3B)
IGNORE_MAX_LENGTH="${IGNORE_MAX_LENGTH:-false}"  # Set to 'true' to disable truncation (for testing beyond model capacity)
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"  # Options: flash_attention_2, sdpa, eager
DEVICE_MAP="${DEVICE_MAP:-auto}"

# Emu3 image tokenizer configuration
EMU3_MIN_PIXELS="${EMU3_MIN_PIXELS:-262144}"     # 512*512
EMU3_MAX_PIXELS="${EMU3_MAX_PIXELS:-1048576}"    # 1024*1024

# Environment configuration
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# SETUP Dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Install task-specific dependencies
echo "📦 Installing task-specific dependencies..."
source "${REPO_ROOT}/examples/install_task_deps.sh" "${TASKS}" "${REPO_ROOT}"

# Install Emu3 model dependencies
echo "📦 Installing Emu3 model dependencies..."
cd "${REPO_ROOT}" || exit 1
pip install -e .[emu3]

# ========== Print Configuration ==========
echo "======================================"
echo "Llama-Emu3 Benchmark Configuration"
echo "======================================"
echo "Model: $PRETRAINED"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "Num Processes: $NUM_PROCESSES"
echo "Max Length: ${MAX_LENGTH:-auto}"
echo "Ignore Max Length: $IGNORE_MAX_LENGTH"
echo "Attention: $ATTN_IMPLEMENTATION"
echo "Device Map: $DEVICE_MAP"
echo "Emu3 Pixels: ${EMU3_MIN_PIXELS}-${EMU3_MAX_PIXELS}"
echo "======================================"

# ========== Build model_args ==========
MODEL_ARGS="pretrained=$PRETRAINED"
MODEL_ARGS="$MODEL_ARGS,tokenizer_path=$TOKENIZER_PATH"
MODEL_ARGS="$MODEL_ARGS,attn_implementation=$ATTN_IMPLEMENTATION"
MODEL_ARGS="$MODEL_ARGS,device_map=$DEVICE_MAP"
MODEL_ARGS="$MODEL_ARGS,emu3_min_pixels=$EMU3_MIN_PIXELS"
MODEL_ARGS="$MODEL_ARGS,emu3_max_pixels=$EMU3_MAX_PIXELS"

# Add optional parameters
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="$MODEL_ARGS,max_length=$MAX_LENGTH"
fi

if [ "$IGNORE_MAX_LENGTH" = "true" ]; then
    MODEL_ARGS="$MODEL_ARGS,ignore_max_length=True"
fi

# ========== Run Evaluation ==========
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --main_process_port=$MAIN_PROCESS_PORT \
    -m lmms_eval \
    --model llama_emu3 \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE"
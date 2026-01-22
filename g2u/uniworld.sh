#!/bin/bash
# UniWorld Model - General Evaluation Script
#
# This script evaluates UniWorld-V1 on any task
# Supports both understanding and generation tasks
#
# Usage:
#   bash uniworld_general.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH] [MASTER_PORT]
#
# Examples:
#   # ChartQA
#   bash /home/aiscuser/lmms-eval/g2u/uniworld.sh "0" "chartqa100" "./logs/chartqa"
#
#   # Uni-MMMU Jigsaw Visual CoT
#   bash /home/aiscuser/lmms-eval/g2u/uniworld.sh "0,1" "uni_mmmu_jigsaw100_visual_cot" "./logs/jigsaw"
#
#   # Uni-MMMU Maze Visual CoT
#   bash uniworld_general.sh "0,1" "uni_mmmu_maze100_visual_cot" "./logs/maze"
#
#   # Multiple tasks (comma-separated, no spaces)
#   bash uniworld_general.sh "0" "chartqa100,mmbench" "./logs/multi"
#
#   # With custom port
#   bash uniworld_general.sh "0" "chartqa100" "./logs/chartqa" "LanguageBind/UniWorld-V1" "29603"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"chartqa100"}
OUTPUT_PATH=${3:-"./logs/uniworld_${TASK}"}
MODEL_PATH=${4:-"LanguageBind/UniWorld-V1"}
MASTER_PORT=${5:-"29602"}
BATCH_SIZE=1

# Check if task includes visual_cot (needs image output)
if [[ "$TASK" == *"visual_cot"* ]]; then
    IMAGE_OUTPUT_DIR="${OUTPUT_PATH}/images"
    MODEL_ARGS="pretrained=${MODEL_PATH},mode=generation,image_output_dir=${IMAGE_OUTPUT_DIR}"
    echo "Visual CoT task detected - mode=generation, images will be saved to: ${IMAGE_OUTPUT_DIR}"
else
    MODEL_ARGS="pretrained=${MODEL_PATH},mode=understanding"
    echo "Understanding task detected - mode=understanding (no image generation)"
fi

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

# ============ Print Configuration ============
echo "======================================"
echo "UniWorld - General Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Model Args:    ${MODEL_ARGS}"
echo "Master Port:   ${MASTER_PORT}"
echo "======================================"
echo ""

# ============ Run Evaluation ============
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model uniworld \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  --verbosity INFO

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "======================================"

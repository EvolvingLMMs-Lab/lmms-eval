#!/bin/bash
# UniWorld Visual CoT - General Evaluation Script
#
# This script evaluates UniWorld with Visual Chain-of-Thought on any task
#
# Usage:
#   bash uniworld_cot.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH] [MASTER_PORT]
#
# Examples:
#   # Uni-MMMU Jigsaw Visual CoT
#   bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0,1" "uni_mmmu_jigsaw100_visual_cot" "./logs/jigsaw_cot"
#
#   # Uni-MMMU Maze Visual CoT
#   bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0,1" "uni_mmmu_maze100_visual_cot" "./logs/maze_cot"
#
#   # All Uni-MMMU CoT tasks
#   bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0,1" "uni_mmmu_cot" "./logs/uni_mmmu_cot"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"uni_mmmu_cot"}
OUTPUT_PATH=${3:-"./logs/uniworld_cot_${TASK}"}
MODEL_PATH=${4:-"LanguageBind/UniWorld-V1"}
MASTER_PORT=${5:-"29700"}
HF_REPO=${6:-""}  # Optional: HF repo for uploading logs (e.g., "username/uniworld-results")
BATCH_SIZE=1

# Model args
MODEL_ARGS="pretrained=${MODEL_PATH}"
if [ -n "$HF_REPO" ]; then
    MODEL_ARGS="${MODEL_ARGS},hf_repo=${HF_REPO},hf_upload=True"
    echo "📤 Hugging Face upload enabled: ${HF_REPO}"
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
echo "UniWorld Visual CoT - Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Master Port:   ${MASTER_PORT}"
if [ -n "$HF_REPO" ]; then
    echo "HF Upload:     ${HF_REPO}"
fi
echo "======================================"
echo ""

# ============ Run Evaluation ============
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model uniworld_visual_cot \
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

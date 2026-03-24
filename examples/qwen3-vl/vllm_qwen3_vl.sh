#!/bin/bash
set -euo pipefail


source /mnt/cpfs/yangyicun/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/cpfs/yangyicun/miniconda3/envs/lmms-eval

MODEL="${MODEL:-/mnt/cpfs/yangyicun/data/model/Qwen3-VL-8B-Instruct}"
# TASKS="${TASKS:-mmmu,seedbench,ocrbench,vizwiz_vqa_val,scienceqa,textvqa_val}"
TASKS="${TASKS:-ocrbench}"


OUTPUT_PATH="${OUTPUT_PATH:-/mnt/cpfs/yangyicun/eval_result}"
VERBOSITY="${VERBOSITY:-DEBUG}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
echo "[INFO] vLLM test (Qwen3-VL)"
echo "[INFO] model=${MODEL} tasks=${TASKS}  batch_size=${BATCH_SIZE} tp=${TENSOR_PARALLEL_SIZE}"
echo "[INFO] output_path=${OUTPUT_PATH}"

python -m lmms_eval \
  --model vllm \
  --model_args "model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

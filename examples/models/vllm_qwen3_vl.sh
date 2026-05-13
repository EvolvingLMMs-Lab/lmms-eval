#!/bin/bash

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-18000000}"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
TASKS="${TASKS:-mme}"
LIMIT="${LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/vllm_qwen3_vl}"
VERBOSITY="${VERBOSITY:-DEBUG}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

echo "[INFO] vLLM test (Qwen3-VL)"
echo "[INFO] model=${MODEL} tasks=${TASKS} limit=${LIMIT} batch_size=${BATCH_SIZE} tp=${TENSOR_PARALLEL_SIZE}"
echo "[INFO] output_path=${OUTPUT_PATH}"

uv run python -m lmms_eval \
  --model vllm \
  --model_args "model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

#!/bin/bash

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
TASKS="${TASKS:-mme}"
LIMIT="${LIMIT:-8}"
WORKER_GPUS="${WORKER_GPUS:-0,1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/async_hf_model}"
VERBOSITY="${VERBOSITY:-DEBUG}"

echo "[INFO] Async HF local workers"
echo "[INFO] model=${MODEL} tasks=${TASKS} limit=${LIMIT} worker_gpus=${WORKER_GPUS}"
echo "[INFO] output_path=${OUTPUT_PATH}"

uv run python -m lmms_eval \
  --model async_hf_model \
  --model_args "pretrained=${MODEL},worker_gpus=${WORKER_GPUS}" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

#!/bin/bash

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
TASKS="${TASKS:-mme}"
LIMIT="${LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/local_qwen3_vl}"
VERBOSITY="${VERBOSITY:-DEBUG}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

echo "[INFO] Local test (Qwen3-VL)"
echo "[INFO] model=${MODEL} tasks=${TASKS} limit=${LIMIT} batch_size=${BATCH_SIZE} device_map=${DEVICE_MAP}"
echo "[INFO] output_path=${OUTPUT_PATH}"

uv run python -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=${MODEL},device_map=${DEVICE_MAP}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

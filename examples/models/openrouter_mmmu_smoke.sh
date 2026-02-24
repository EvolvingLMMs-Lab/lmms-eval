#!/usr/bin/env bash

set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_VERSION="${MODEL_VERSION:-openai/gpt-4.1-mini}"
TASKS="${TASKS:-mmmu_val}"
LIMIT="${LIMIT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/openrouter_mmmu_smoke}"
VERBOSITY="${VERBOSITY:-INFO}"

echo "[INFO] OpenRouter MMMU smoke"
echo "[INFO] model=${MODEL_VERSION} tasks=${TASKS} limit=${LIMIT} batch_size=${BATCH_SIZE}"
echo "[INFO] output_path=${OUTPUT_PATH}"

uv run python -m lmms_eval \
  --model openai \
  --model_args "model_version=${MODEL_VERSION}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --process_with_media \
  --verbosity "${VERBOSITY}"

echo "[INFO] Done. Check samples under: ${OUTPUT_PATH}"

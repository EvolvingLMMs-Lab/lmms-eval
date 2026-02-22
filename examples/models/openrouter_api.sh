#!/bin/bash

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set in environment}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"

MODEL_VERSION="${MODEL_VERSION:-openai/gpt-4.1-mini}"
TASKS="${TASKS:-mme}"
LIMIT="${LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/openrouter_api}"
VERBOSITY="${VERBOSITY:-DEBUG}"

echo "[INFO] OpenRouter API test"
echo "[INFO] model=${MODEL_VERSION} tasks=${TASKS} limit=${LIMIT} batch_size=${BATCH_SIZE}"
echo "[INFO] output_path=${OUTPUT_PATH}"

uv run python -m lmms_eval \
  --model openai_compatible \
  --model_args "model_version=${MODEL_VERSION}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

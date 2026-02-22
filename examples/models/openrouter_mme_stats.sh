#!/bin/bash

set -euo pipefail

# OpenRouter + MME quick test (simple version)
# - Default uses all samples (`LIMIT=-1`)
# - Stability metrics appear when `REPEATS > 1`

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set in environment}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"

MODEL_VERSION="${MODEL_VERSION:-mistralai/ministral-3b-2512}"
TASKS="${TASKS:-mme}"
REPEATS="${REPEATS:-1}"
LIMIT="${LIMIT:--1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/openrouter_mme_stats/}"
VERBOSITY="${VERBOSITY:-INFO}"

echo "[INFO] OpenRouter MME test"
echo "[INFO] model=${MODEL_VERSION} tasks=${TASKS} repeats=${REPEATS} limit=${LIMIT}"
echo "[INFO] output_path=${OUTPUT_PATH}"

python3 -m lmms_eval \
  --model openai_compatible \
  --model_args "model_version=${MODEL_VERSION}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --repeats "${REPEATS}" \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

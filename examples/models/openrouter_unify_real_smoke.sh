#!/usr/bin/env bash

set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"

USE_HF_TOKEN="${USE_HF_TOKEN:-1}"

if [[ "${USE_HF_TOKEN}" == "1" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_HUB_DISABLE_IMPLICIT_TOKEN=0
    echo "[INFO] HF token provided via HF_TOKEN. Using authenticated dataset access."
  else
    echo "[WARN] HF_TOKEN is not set. Falling back to anonymous access."
    export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  fi
else
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  echo "[INFO] USE_HF_TOKEN=0, forcing anonymous dataset access."
fi

MODEL_VERSION="${MODEL_VERSION:-openai/gpt-4.1-mini}"
TASKS="${TASKS:-realunify_mental_tracking,uni_mmmu_jigsaw}"
LIMIT="${LIMIT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/openrouter_unify_real_smoke}"
VERBOSITY="${VERBOSITY:-INFO}"

echo "[INFO] OpenRouter Unify real-dataset smoke"
echo "[INFO] model=${MODEL_VERSION} tasks=${TASKS} limit=${LIMIT}"
echo "[INFO] output_path=${OUTPUT_PATH}"
echo "[INFO] HF_HUB_DISABLE_IMPLICIT_TOKEN=${HF_HUB_DISABLE_IMPLICIT_TOKEN}"

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

echo "[INFO] Done. Real-dataset smoke logs saved under: ${OUTPUT_PATH}"

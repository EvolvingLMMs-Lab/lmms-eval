#!/usr/bin/env bash

set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_VERSION="${MODEL_VERSION:-google/gemini-2.5-flash-image}"
TASKS="${TASKS:-realunify_gen_smoke}"
LIMIT="${LIMIT:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/openrouter_unify_image_smoke}"
IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-./logs/openrouter_unify_image_outputs}"

echo "[INFO] OpenRouter Unify image smoke"
echo "[INFO] model=${MODEL_VERSION} tasks=${TASKS} limit=${LIMIT}"
echo "[INFO] output_path=${OUTPUT_PATH} image_output_dir=${IMAGE_OUTPUT_DIR}"

uv run python -m lmms_eval \
  --model openrouter_image_gen \
  --model_args "model=${MODEL_VERSION},output_dir=${IMAGE_OUTPUT_DIR},num_concurrent=1,max_tokens=4096,image_size=1024x1024" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --process_with_media \
  --verbosity INFO

echo "[INFO] Done. Generated images under: ${IMAGE_OUTPUT_DIR}"

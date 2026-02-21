#!/bin/bash

set -euo pipefail

LIMIT="${LIMIT:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/minerva_dummy_video_reader/}"
VERBOSITY="${VERBOSITY:-INFO}"

echo "[INFO] MINERVA dummy video-reader benchmark"
echo "[INFO] tasks=minerva limit=${LIMIT} batch_size=${BATCH_SIZE}"

uv run --with pylance --with pyarrow python -m lmms_eval \
    --model dummy_video_reader \
    --model_args "read_bytes=65536,response=A,allow_remote=false,fail_on_missing=true" \
    --tasks minerva \
    --batch_size "${BATCH_SIZE}" \
    --limit "${LIMIT}" \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --verbosity "${VERBOSITY}"

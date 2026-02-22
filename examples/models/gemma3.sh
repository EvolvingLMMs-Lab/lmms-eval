#!/bin/bash
# Run and exactly reproduce gemma3 results!
# mme as an example

NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PORT="${MAIN_PORT:-12345}"
MODEL_ID="${MODEL_ID:-google/gemma-3-4b-it}"
TASKS="${TASKS:-mmmu_val,ai2d,mathvista_testmini}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/}"

accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PORT}" -m lmms_eval \
  --model gemma3 \
  --model_args "pretrained=${MODEL_ID}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" --output_path "${OUTPUT_PATH}"
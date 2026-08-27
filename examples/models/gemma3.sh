#!/bin/bash
# Run and exactly reproduce gemma3 results!
# videomme_v2 as an example

NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PORT="${MAIN_PORT:-12345}"
MODEL_ID="${MODEL_ID:-google/gemma-3-4b-it}"
TASKS="${TASKS:-mmmu_val,ai2d,mathvista_testmini,videomme_v2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"
INTERLEAVE_VISUALS="${INTERLEAVE_VISUALS:-False}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
# LMMS_VIDEO_DECODE_BACKEND="${LMMS_VIDEO_DECODE_BACKEND:-torchcodec}"
# export LMMS_VIDEO_DECODE_BACKEND

accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PORT}" -m lmms_eval \
  --model gemma3 \
  --model_args "pretrained=${MODEL_ID},max_num_frames=${MAX_NUM_FRAMES},interleave_visuals=${INTERLEAVE_VISUALS}" \
  --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS},temperature=${TEMPERATURE},top_p=${TOP_P},top_k=${TOP_K}" \
  --tasks "${TASKS}" \
  --log_samples \
  --batch_size "${BATCH_SIZE}" --output_path "${OUTPUT_PATH}"
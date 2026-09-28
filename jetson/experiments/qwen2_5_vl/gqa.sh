#!/usr/bin/env bash
# GQA testdev-balanced (12,578 short-answer questions on 398 images, exact match), 3B and 7B.
# Needs dataset:lmms-lab-encoder/GQA:testdev_balanced_instructions and :testdev_balanced_images (download_assets.sh).
# SMOKE=1: 8 samples each. Results: $RESULTS_DIR/<model>/gqa/...
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
MATRIX=(
  hf 3b hf 7b
  vllm 3b vllm 3b-awq vllm 7b vllm 7b-awq
  llamacpp 3b-q8_0 llamacpp 3b-q4_k_m llamacpp 7b-q8_0 llamacpp 7b-q4_k_m
  trt_edgellm 3b-fp16 trt_edgellm 3b-fp8 trt_edgellm 3b-int4_awq
)
[ "${SMOKE:-0}" = 1 ] && exec "$REPO/jetson/scripts/run_matrix.sh" gqa 8 -- "${MATRIX[@]}"
exec "$REPO/jetson/scripts/run_matrix.sh" gqa "" -- "${MATRIX[@]}"

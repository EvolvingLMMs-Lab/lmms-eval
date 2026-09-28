#!/usr/bin/env bash
# Main comparison: MME (2,374 yes/no questions) for every framework and precision, 3B and 7B.
# Smoke test first (SMOKE=1: 8 samples each), then the full runs. Results: $RESULTS_DIR/<model>/mme/...
# Usage: jetson/experiments/qwen2_5_vl/mme.sh            (TensorRT Edge-LLM needs its engines, see docs/tensorrt-edgellm.md)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
MATRIX=(
  hf 3b hf 7b
  vllm 3b vllm 3b-awq vllm 7b vllm 7b-awq
  llamacpp 3b-q8_0 llamacpp 3b-q4_k_m llamacpp 7b-q8_0 llamacpp 7b-q4_k_m
  trt_edgellm 3b-fp16 trt_edgellm 3b-fp8 trt_edgellm 3b-int4_awq
)
[ "${SMOKE:-0}" = 1 ] && exec "$REPO/jetson/scripts/run_matrix.sh" mme 8 -- "${MATRIX[@]}"
exec "$REPO/jetson/scripts/run_matrix.sh" mme "" -- "${MATRIX[@]}"

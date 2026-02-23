#!/usr/bin/env bash
# api_gen model backend - video generation via fal.ai API
#
# Supports: Wan, LTX, HunyuanVideo model families
#
# Prerequisites:
#   export FAL_KEY="your-fal-api-key"  # Get one at https://fal.ai/dashboard/keys
#
# Usage examples:
#   # Default: Wan 2.6 T2V
#   bash examples/models/api_videogen.sh
#
#   # Run one VBench v1 dimension
#   TASKS=vbench_human_action bash examples/models/api_videogen.sh
#
#   # Run all VBench 2.0 dimensions
#   TASKS=vbench2 bash examples/models/api_videogen.sh
#
#   # Run all VBench dimensions (v1 + v2)
#   TASKS=vbench_all bash examples/models/api_videogen.sh
#
#   # LTX Video
#   FAL_MODEL=fal-ai/ltx-video bash examples/models/api_videogen.sh
#
#   # HunyuanVideo
#   FAL_MODEL=fal-ai/hunyuan-video bash examples/models/api_videogen.sh
#
#   # HunyuanVideo v1.5
#   FAL_MODEL=fal-ai/hunyuan-video-v1.5/text-to-video bash examples/models/api_videogen.sh

set -euo pipefail

FAL_MODEL="${FAL_MODEL:-wan/v2.6/text-to-video}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/api_gen_videos}"
LIMIT="${LIMIT:-4}"
CONCURRENCY="${CONCURRENCY:-4}"
RESOLUTION="${RESOLUTION:-720p}"
DURATION="${DURATION:-5}"
TASKS="${TASKS:-vbench}"

echo "[INFO] api_gen video generation"
echo "[INFO] model=${FAL_MODEL}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] limit=${LIMIT}"
echo "[INFO] concurrency=${CONCURRENCY}"

python -m lmms_eval \
  --model api_gen \
  --model_args "model=${FAL_MODEL},output_dir=${OUTPUT_DIR},num_concurrent=${CONCURRENCY},resolution=${RESOLUTION},duration=${DURATION}" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --limit "${LIMIT}" \
  --log_samples \
  --verbosity INFO

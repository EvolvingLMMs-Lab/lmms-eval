#!/usr/bin/env bash

# Adaptive concurrency benchmark for OpenRouter API models.
# See openrouter_molmo_throughput_compare.sh for baseline/static comparisons.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

MODEL_VERSION="bytedance-seed/seed-1.6-flash"
TASKS="mme"
LIMIT="${1:-40}"
OUTPUT_DIR="./logs/openrouter_molmo_throughput/adaptive"

# num_concurrent=16 reaches peak throughput for most OpenRouter endpoints.
# Only non-default adaptive params are listed; defaults: min=1, target_latency=15s, failure_threshold=0.05
ADAPTIVE_ARGS="num_concurrent=16,adaptive_concurrency=true,adaptive_max_concurrency=64,adaptive_increase_step=0.15,adaptive_decrease_factor=0.75,max_retries=1"

mkdir -p "$OUTPUT_DIR"

START_NS=$(date +%s%N)
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "model_version=$MODEL_VERSION,$ADAPTIVE_ARGS" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --limit "$LIMIT" \
    --output_path "${OUTPUT_DIR}/results" \
    --log_samples 2>&1 | tee "${OUTPUT_DIR}/run.log"
END_NS=$(date +%s%N)

WALL_S=$(awk -v s="$START_NS" -v e="$END_NS" 'BEGIN{printf "%.3f",(e-s)/1e9}')
RPS=$(awk -v l="$LIMIT" -v w="$WALL_S" 'BEGIN{if(w>0) printf "%.3f",l/w; else print 0}')
printf "wall=%.1fs  rps=%s\n" "$WALL_S" "$RPS"

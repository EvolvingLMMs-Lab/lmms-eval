#!/usr/bin/env bash

# OpenRouter Molmo Throughput Baseline vs static concurrency sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

MODEL_VERSION="bytedance-seed/seed-1.6-flash"
TASKS="mme"
LIMIT="${1:-40}"
BATCH_SIZE="1"
VERBOSITY="INFO"
MODEL_TIMEOUT=10
MODEL_MAX_RETRIES=1
OUTPUT_BASE="./logs/openrouter_molmo_throughput"
CONCURRENCIES=(2 4 8 16 24)

RUN_NAME="static_concurrency"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"
SUMMARY_FILE="${RUN_DIR}/summary.csv"
mkdir -p "$RUN_DIR"

echo "mode,concurrency,limit,wall_time_s,requests_per_sec,log_path" > "$SUMMARY_FILE"

for CONCURRENCY in "${CONCURRENCIES[@]}"; do
  CONCURRENCY_DIR="${RUN_DIR}/c${CONCURRENCY}"
  mkdir -p "$CONCURRENCY_DIR"

  START_NS=$(date +%s%N)
  python3 -m lmms_eval \
      --model openai_compatible \
      --model_args model_version=$MODEL_VERSION,num_concurrent=$CONCURRENCY,timeout=$MODEL_TIMEOUT,max_retries=$MODEL_MAX_RETRIES,adaptive_concurrency=false \
      --tasks "$TASKS" \
      --batch_size "$BATCH_SIZE" \
      --limit "$LIMIT" \
      --output_path "${CONCURRENCY_DIR}/results" \
      --verbosity "$VERBOSITY" \
      --log_samples 2>&1 | tee "${CONCURRENCY_DIR}/run.log"
  END_NS=$(date +%s%N)

  WALL_TIME_S=$(awk -v start="$START_NS" -v end="$END_NS" 'BEGIN { printf "%.6f", (end - start) / 1000000000 }')
  REQ_PER_S=$(awk -v limit="$LIMIT" -v wall="$WALL_TIME_S" 'BEGIN { if (wall > 0) printf "%.6f", limit / wall; else print "0" }')

  echo "static,$CONCURRENCY,$LIMIT,$WALL_TIME_S,$REQ_PER_S,${CONCURRENCY_DIR}/run.log" >> "$SUMMARY_FILE"
done

printf "STATIC_CONCURRENCY done: %s\n" "$SUMMARY_FILE"

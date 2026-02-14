#!/usr/bin/env bash

# OpenRouter Molmo Throughput Baseline (single concurrency)

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
MODEL_TIMEOUT=10
MODEL_MAX_RETRIES=1
TASKS="mme"
LIMIT="${1:-40}"
BATCH_SIZE="1"
VERBOSITY="INFO"
OUTPUT_BASE="./logs/openrouter_molmo_throughput"

RUN_NAME="baseline"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"
SUMMARY_FILE="${RUN_DIR}/summary.csv"
mkdir -p "$RUN_DIR"

START_NS=$(date +%s%N)
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=$MODEL_VERSION,num_concurrent=1,timeout=$MODEL_TIMEOUT,max_retries=$MODEL_MAX_RETRIES,adaptive_concurrency=false \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --output_path "${RUN_DIR}/results" \
    --verbosity "$VERBOSITY" \
    --log_samples 2>&1 | tee "${RUN_DIR}/run.log"
END_NS=$(date +%s%N)

WALL_TIME_S=$(awk -v start="$START_NS" -v end="$END_NS" 'BEGIN { printf "%.6f", (end - start) / 1000000000 }')
REQ_PER_S=$(awk -v limit="$LIMIT" -v wall="$WALL_TIME_S" 'BEGIN { if (wall > 0) printf "%.6f", limit / wall; else print "0" }')

cat > "$SUMMARY_FILE" <<EOF
mode,concurrency,limit,wall_time_s,requests_per_sec,log_path
baseline,1,$LIMIT,$WALL_TIME_S,$REQ_PER_S,${RUN_DIR}/run.log
EOF

printf "BASELINE done: %s\n" "$SUMMARY_FILE"
printf "requests_per_sec=%s\n" "$REQ_PER_S"

#!/usr/bin/env bash

# Throughput comparison: baseline vs static concurrency vs adaptive concurrency
# Usage: bash openrouter_molmo_throughput_compare.sh [LIMIT]
#   LIMIT defaults to 40. Use openrouter_molmo_adaptive.sh for production runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

MODEL_VERSION="bytedance-seed/seed-1.6-flash"
TASKS="mme"
LIMIT="${1:-40}"
TIMEOUT=10
MAX_RETRIES=1
OUTPUT_BASE="./logs/openrouter_molmo_throughput"
COMPARISON="${OUTPUT_BASE}/throughput_comparison.csv"

echo "mode,concurrency,limit,wall_time_s,requests_per_sec" > "$COMPARISON"

run_benchmark() {
  local mode="$1" concurrency="$2" extra_args="${3:-}"
  local run_dir="${OUTPUT_BASE}/${mode}_c${concurrency}"
  mkdir -p "$run_dir"

  local start_ns=$(date +%s%N)
  python3 -m lmms_eval \
      --model openai_compatible \
      --model_args "model_version=$MODEL_VERSION,num_concurrent=$concurrency,timeout=$TIMEOUT,max_retries=$MAX_RETRIES${extra_args}" \
      --tasks "$TASKS" \
      --batch_size 1 \
      --limit "$LIMIT" \
      --output_path "${run_dir}/results" \
      --verbosity INFO \
      --log_samples 2>&1 | tee "${run_dir}/run.log"
  local end_ns=$(date +%s%N)

  local wall=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.3f",(e-s)/1e9}')
  local rps=$(awk -v l="$LIMIT" -v w="$wall" 'BEGIN{if(w>0) printf "%.3f",l/w; else print 0}')
  echo "$mode,$concurrency,$LIMIT,$wall,$rps" >> "$COMPARISON"
  printf "[%s] c=%s  wall=%.1fs  rps=%s\n" "$mode" "$concurrency" "$wall" "$rps"
}

# 1. Baseline (sequential)
run_benchmark "baseline" 1 ",adaptive_concurrency=false"

# 2. Static concurrency sweep
for c in 2 4 8 16 24; do
  run_benchmark "static" "$c" ",adaptive_concurrency=false"
done

# 3. Adaptive concurrency
run_benchmark "adaptive" 16 ",adaptive_concurrency=true,adaptive_min_concurrency=1,adaptive_max_concurrency=64,adaptive_target_latency_s=15.0,adaptive_increase_step=0.15,adaptive_decrease_factor=0.75,adaptive_failure_threshold=0.05"

echo ""
echo "=== Results ==="
column -t -s, "$COMPARISON"

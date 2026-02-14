#!/usr/bin/env bash

# Orchestrated benchmark and throughput comparison for OpenRouter Molmo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}"

LIMIT="${1:-40}"
OUTPUT_BASE="./logs/openrouter_molmo_throughput"
mkdir -p "$OUTPUT_BASE"

bash "${SCRIPT_DIR}/openrouter_molmo_baseline.sh" "$LIMIT"
bash "${SCRIPT_DIR}/openrouter_molmo_static_concurrency.sh" "$LIMIT"
bash "${SCRIPT_DIR}/openrouter_molmo_adaptive.sh" "$LIMIT"

BASELINE_SUMMARY="${OUTPUT_BASE}/baseline/summary.csv"
STATIC_SUMMARY="${OUTPUT_BASE}/static_concurrency/summary.csv"
ADAPTIVE_SUMMARY="${OUTPUT_BASE}/adaptive/summary.csv"
COMPARISON_SUMMARY="${OUTPUT_BASE}/throughput_comparison.csv"

BASELINE_RPS="$(awk -F, 'NR==2 {print $5}' "$BASELINE_SUMMARY")"

if [[ -z "$BASELINE_RPS" || "$BASELINE_RPS" == "0" ]]; then
  echo "baseline requests_per_sec not found or zero, aborting comparison"
  exit 1
fi

echo "run_type,concurrency,requests_per_sec,wall_time_s,improvement_pct,log_path" > "$COMPARISON_SUMMARY"

echo "baseline,1,$BASELINE_RPS,0,0,${OUTPUT_BASE}/baseline/run.log" >> "$COMPARISON_SUMMARY"

dedupe_summary() {
  local summary_path="$1"
  awk -F, '
    NR > 1 {
      key = $1 SUBSEP $2;
      if (!(key in seen_order)) {
        order[++n] = key;
        seen_order[key] = n;
        rows[key] = $0;
        rates[key] = $5;
      } else if ($5 + 0 > rates[key] + 0) {
        rows[key] = $0;
        rates[key] = $5;
      }
    }
    END {
      for (i = 1; i <= n; i++) {
        print rows[order[i]];
      }
    }
  ' "$summary_path"
}

for SUMMARY_PATH in "$STATIC_SUMMARY" "$ADAPTIVE_SUMMARY"; do
  while IFS=, read -r MODE CONCURRENCY LIMIT_FIELD WALL REQUESTS_PER_SEC LOG_PATH; do
    # skip malformed lines
    if [[ "$MODE" == "mode" || "$MODE" == "" ]]; then
      continue
    fi
    IMPROVEMENT_PCT="$(awk -v base="$BASELINE_RPS" -v current="$REQUESTS_PER_SEC" 'BEGIN { printf "%.2f", (current / base - 1) * 100 }')"
    echo "$MODE,$CONCURRENCY,$REQUESTS_PER_SEC,$WALL,$IMPROVEMENT_PCT,$LOG_PATH" >> "$COMPARISON_SUMMARY"
  done < <(dedupe_summary "$SUMMARY_PATH")
done

echo "Comparison saved to: $COMPARISON_SUMMARY"
cat "$COMPARISON_SUMMARY"

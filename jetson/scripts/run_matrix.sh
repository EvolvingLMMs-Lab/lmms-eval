#!/usr/bin/env bash
# Run several framework/model combinations one after another, then refresh results/SUMMARY.md.
# A failed run is reported and skipped; the rest continue.
#
# Usage: jetson/scripts/run_matrix.sh [tasks=mme] [limit] [-- <framework> <size-precision> ...]
#   jetson/scripts/run_matrix.sh mme                     # default matrix below
#   jetson/scripts/run_matrix.sh mme 8                   # smoke test of every combination
#   jetson/scripts/run_matrix.sh mme "" -- vllm 3b llamacpp 3b-q8_0
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/jetson/frameworks/common.sh"  # RESULTS_DIR
TASKS=${1:-mme}
LIMIT=${2:-}
shift $(($# < 2 ? $# : 2))
[ "${1:-}" = "--" ] && shift

if [ $# -gt 0 ]; then
  MATRIX=("$@")
else
  MATRIX=(
    hf 3b          hf 7b
    vllm 3b        vllm 3b-awq        vllm 7b        vllm 7b-awq
    llamacpp 3b-q8_0 llamacpp 3b-q4_k_m llamacpp 7b-q8_0 llamacpp 7b-q4_k_m
  )
fi

failed=()
for ((i = 0; i < ${#MATRIX[@]}; i += 2)); do
  framework=${MATRIX[i]} spec=${MATRIX[i + 1]}
  echo "=== $(date +%H:%M:%S) $framework $spec $TASKS ${LIMIT:+limit=$LIMIT}"
  if ! "$REPO/jetson/scripts/run_eval.sh" "$framework" "$spec" "$TASKS" $LIMIT >/dev/null 2>&1; then
    echo "    FAILED (see the newest run dir under $RESULTS_DIR/*/*/$framework-*/)"
    failed+=("$framework $spec")
  fi
done

python3 "$REPO/jetson/scripts/summarize.py" ${LIMIT:+--include-smoke} >/dev/null
echo "summary: $REPO/$RESULTS_DIR/SUMMARY.md"
[ ${#failed[@]} -eq 0 ] || { printf 'failed: %s\n' "${failed[@]}"; exit 1; }

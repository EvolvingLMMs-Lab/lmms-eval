#!/usr/bin/env bash

set -euo pipefail

# OpenRouter + MME statistics probe
#
# What this script gives you from one run:
# - Standard stderr (`*_stderr,*`)
# - CLT stderr (`*_stderr_clt,*`)
# - Clustered stderr (`*_stderr_clustered,*`, when task defines clusters)
# - Stability metrics (`*_expected_accuracy,*`, `*_consensus_accuracy,*`,
#   `*_internal_variance,*`, `*_consistency_rate,*`) when NUM_SAMPLES > 1
# - Optional paired baseline stats (`paired_ci_*`, `paired_pvalue`) if BASELINE is set

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}"

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "ERROR: OPENAI_API_KEY or OPENROUTER_API_KEY must be set." >&2
  exit 1
fi

MODEL_VERSION="${MODEL_VERSION:-allenai/molmo-2-8b}"
TASKS="${TASKS:-mme}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
LIMIT="${LIMIT:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VERBOSITY="${VERBOSITY:-INFO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/openrouter_mme_stats}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_ID}"

echo "[INFO] OpenRouter MME statistics test"
echo "[INFO] model=${MODEL_VERSION} tasks=${TASKS} num_samples=${NUM_SAMPLES} limit=${LIMIT}"
echo "[INFO] output_path=${OUTPUT_PATH}"

cmd=(
  uv run -m lmms_eval
  --model openai_compatible
  --model_args "model_version=${MODEL_VERSION},base_url=${OPENAI_API_BASE}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --num_samples "${NUM_SAMPLES}"
  --limit "${LIMIT}"
  --output_path "${OUTPUT_PATH}"
  --log_samples
  --verbosity "${VERBOSITY}"
)

if [[ -n "${BASELINE:-}" ]]; then
  echo "[INFO] baseline=${BASELINE}"
  cmd+=(--baseline "${BASELINE}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] Command:"
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

"${cmd[@]}"

python3 - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
result_files = sorted(output_path.rglob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if not result_files:
    raise SystemExit(f"[ERROR] No *_results.json found under: {output_path}")

result_file = result_files[0]
print(f"[INFO] result_file={result_file}")

with result_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", {})
if not isinstance(results, dict):
    raise SystemExit("[ERROR] Unexpected results format: missing dict at key 'results'.")

keywords = (
    "_stderr,",
    "_stderr_clt,",
    "_stderr_clustered,",
    "_expected_accuracy,",
    "_consensus_accuracy,",
    "_internal_variance,",
    "_consistency_rate,",
    "paired_ci_lower",
    "paired_ci_upper",
    "paired_pvalue",
    "samples",
)

print("\n[STATS] Extracted metrics")
found = False
for task_name in sorted(results.keys()):
    task_result = results.get(task_name, {})
    if not isinstance(task_result, dict):
        continue
    task_lines = []
    for key, value in sorted(task_result.items()):
        if any(k in key for k in keywords):
            task_lines.append(f"  - {key}: {value}")
    if task_lines:
        found = True
        print(f"\n[{task_name}]")
        for line in task_lines:
            print(line)

if not found:
    print("[WARN] No stats-related keys were found in results. Check task output format and flags.")
PY

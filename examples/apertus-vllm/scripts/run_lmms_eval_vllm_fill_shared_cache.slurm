#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=lmms-fill-shared-sqlite
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/apertus-vllm-lmms-eval-prod.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/logs/lmms_fill_shared_sqlite_wrapper_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/logs/lmms_fill_shared_sqlite_wrapper_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/run_lmms_eval_vllm_prod.slurm}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Wrapper: run_lmms_eval_vllm_fill_shared_cache.slurm

Purpose:
  Fill/populate the shared SQLite image-token cache artifact.

Pinned cache flags:
  --enable-image-token-cache true
  --image-token-cache-preload 0

Pass all regular lmms-eval launcher flags after this script.
EOF
  echo
  bash "${LAUNCHER}" --help
  exit 0
fi

exec bash "${LAUNCHER}" \
  "$@" \
  --enable-image-token-cache true \
  --image-token-cache-preload 0

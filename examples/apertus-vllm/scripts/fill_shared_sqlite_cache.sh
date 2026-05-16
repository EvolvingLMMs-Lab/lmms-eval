#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash fill_shared_sqlite_cache.sh [models.txt]
#
# Purpose:
#   Populate/extend the shared SQLite image-token cache for new benchmarks/tasks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_LIST="${1:-${SCRIPT_DIR}/models.txt}"

if [[ ! -f "${MODEL_LIST}" ]]; then
  echo "Model list file not found: ${MODEL_LIST}"
  exit 1
fi

SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/run_lmms_eval_vllm_fill_shared_cache.slurm}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct}"
TASKS_CSV="${TASKS_CSV:-gqa,realworldqa,seedbench,mmstar,ocrbench_v2,textvqa_val,docvqa_val,vqav2_val,ocrbench,infovqa_val,chartqa,refcoco,pope,mme,ai2d,mmmu_val,mathvision_test,mathvision_testmini,VisualPuzzles_direct,countbench,embspatial,screenspot,osworld_g}"
IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_DIR:-${REPO_ROOT}/cache}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-${LOG_DIR}/lmms_fill_shared_sqlite_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-${LOG_DIR}/lmms_fill_shared_sqlite_%j.err}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-5}"

mkdir -p "${LOG_DIR}" "${IMAGE_TOKEN_CACHE_DIR}"

IFS=',' read -ra TASK_ARRAY <<< "${TASKS_CSV}"

while IFS= read -r MODEL_PATH || [[ -n "${MODEL_PATH}" ]]; do
  [[ -z "${MODEL_PATH}" ]] && continue
  [[ "${MODEL_PATH}" =~ ^[[:space:]]*# ]] && continue

  MODEL_PATH="$(echo "${MODEL_PATH}" | xargs)"
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "WARNING: model path does not exist, skipping:"
    echo "  ${MODEL_PATH}"
    continue
  fi

  MODEL_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
  if [[ "$(basename "${MODEL_PATH}")" != "HF" ]]; then
    MODEL_NAME="$(basename "${MODEL_PATH}")"
  fi

  for TASK in "${TASK_ARRAY[@]}"; do
    TASK="$(echo "${TASK}" | xargs)"
    [[ -z "${TASK}" ]] && continue

    echo "Submitting shared-cache fill job:"
    echo "  MODEL_PATH=${MODEL_PATH}"
    echo "  MODEL_NAME=${MODEL_NAME}"
    echo "  TASK=${TASK}"
    echo "  NUM_PROCESSES=${NUM_PROCESSES}"

    sbatch \
      --output "${SBATCH_OUTPUT}" \
      --error "${SBATCH_ERROR}" \
      "${SLURM_SCRIPT}" \
      --model-path "${MODEL_PATH}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --tasks "${TASK}" \
      --num-processes "${NUM_PROCESSES}" \
      --image-token-cache-dir "${IMAGE_TOKEN_CACHE_DIR}" \
      --image-token-cache-collision-guard 0 \
      --batch-size "${BATCH_SIZE}"

    # sleep "${SUBMIT_SLEEP_SECONDS}"
    echo
  done
done < "${MODEL_LIST}"

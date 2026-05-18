#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash eval_using_local_sqlite_copy.sh models.txt
#
# Purpose:
#   Evaluate using an already-populated shared SQLite cache.
#   Uses node-local cache copy + preload + readonly + no write-misses.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_LIST="${1:-}"

if [[ -z "${MODEL_LIST}" ]]; then
  echo "Usage: bash $0 models.txt"
  exit 1
fi

if [[ ! -f "${MODEL_LIST}" ]]; then
  echo "Model list file not found: ${MODEL_LIST}"
  exit 1
fi

SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/run_lmms_eval_vllm_eval_local_copy.slurm}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct}"
TASKS_CSV="${TASKS_CSV:-gqa,realworldqa,seedbench,mmstar,ocrbench_v2,textvqa_val,docvqa_val,vqav2_val,ocrbench,infovqa_val,chartqa,refcoco,pope,mme,ai2d,mmmu_val,mathvision_test,mathvision_testmini,VisualPuzzles_direct,countbench,embspatial,screenspot,osworld_g}"
IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_DIR:-${REPO_ROOT}/cache}"
IMAGE_TOKEN_CACHE_LOCAL_BASE="${IMAGE_TOKEN_CACHE_LOCAL_BASE:-/tmp/${USER}/apertus_image_token_cache}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-${LOG_DIR}/lmms_eval_read_only_sqlite_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-${LOG_DIR}/lmms_eval_read_only_sqlite_%j.err}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-5}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lmms-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-rkreft-eth-z-rich}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-eval}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-}"
WANDB_ARGS="${WANDB_ARGS:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

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

    echo "Submitting local-copy eval job:"
    echo "  MODEL_PATH=${MODEL_PATH}"
    echo "  MODEL_NAME=${MODEL_NAME}"
    echo "  TASK=${TASK}"
    echo "  NUM_PROCESSES=${NUM_PROCESSES}"

    WANDB_GROUP="${WANDB_GROUP_PREFIX}${MODEL_NAME}"

    sbatch \
      --output "${SBATCH_OUTPUT}" \
      --error "${SBATCH_ERROR}" \
      "${SLURM_SCRIPT}" \
      --model-path "${MODEL_PATH}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --tasks "${TASK}" \
      --num-processes "${NUM_PROCESSES}" \
      --enable-image-token-cache true \
      --image-token-cache-dir "${IMAGE_TOKEN_CACHE_DIR}" \
      --image-token-cache-collision-guard 0 \
      --image-token-cache-local-copy 0 \
      --image-token-cache-local-base "${IMAGE_TOKEN_CACHE_LOCAL_BASE}" \
      --image-token-cache-preload 1 \
      --image-token-cache-readonly 1 \
      --image-token-cache-write-misses 0 \
      --enable-wandb "${ENABLE_WANDB}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-job-type "${WANDB_JOB_TYPE}" \
      --wandb-group "${WANDB_GROUP}" \
      --wandb-log-samples "${WANDB_LOG_SAMPLES}" \
      --wandb-args "${WANDB_ARGS}" \
      --wandb-api-key "${WANDB_API_KEY}" \
      --batch-size "${BATCH_SIZE}"

    # sleep "${SUBMIT_SLEEP_SECONDS}"
    echo
  done
done < "${MODEL_LIST}"

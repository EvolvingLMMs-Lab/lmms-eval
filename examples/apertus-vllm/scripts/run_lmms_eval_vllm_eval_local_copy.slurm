#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=lmms-eval-local-sqlite-copy
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/apertus-vllm-lmms-eval-prod.toml
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/logs/lmms_eval_local_sqlite_copy_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/logs/lmms_eval_local_sqlite_copy_%j.err



# How to run example
# sbatch /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/scripts/run_lmms_eval_vllm_eval_local_copy.slurm \
#     --model-path /capstor/store/.../HF \
#     --tokenizer-path /capstor/store/.../tokenizer/apertus_emu3.5_instruct \
#     --tasks pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa \
#     --hf-home /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/cache/hf \
#     --output-path /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/results/run1 \
#     --num-processes 4 \
#     --gpu-memory-utilization 0.6 \
set -euo pipefail

REPO_ROOT="/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval"
CACHE_ROOT="${REPO_ROOT}/cache"



########################################
# Required runtime configuration
########################################
MODEL_PATH="/path/to/apertus/model-or-hf-id"
TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct"
OUTPUT_PATH="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/logs"
HF_HOME_PATH="${HF_HOME:-${CACHE_ROOT}/hf}"
NLTK_DATA_PATH="${NLTK_DATA:-${CACHE_ROOT}/nltk_data}"
IMAGE_TOKEN_CACHE_DIR="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR:-${CACHE_ROOT}}"
XDG_CACHE_HOME_PATH="${XDG_CACHE_HOME:-${CACHE_ROOT}/.cache}"
VLLM_CACHE_ROOT_PATH="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm_cache}"
LMMS_EVAL_MODELS_CACHE_PATH="${LMMS_EVAL_MODELS_CACHE:-${CACHE_ROOT}/models_cache}"

########################################
# Evaluation configuration
########################################
TASKS="pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
BATCH_SIZE=1
NUM_PROCESSES=4
MAIN_PROCESS_PORT=29541
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=49152
HF_OVERRIDES='{"max_position_embeddings":131072}'
TRUST_REMOTE_CODE=False
ENABLE_PREFIX_CACHING=true
ENFORCE_EAGER=false


OFFLINE_DATASETS=false
LIMIT=""
GEN_KWARGS=""
CHAT_TEMPLATE=""
EXTRA_MODEL_ARGS=""
SEED=1234
EVAL_SEED="${SEED}"
ENABLE_IMAGE_TOKEN_CACHE=true
IMAGE_TOKEN_CACHE_COLLISION_GUARD="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD:-0}"
IMAGE_TOKEN_CACHE_LOCAL_COPY="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_COPY:-1}"
IMAGE_TOKEN_CACHE_LOCAL_BASE="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_BASE:-/tmp/${USER}/apertus_image_token_cache}"
IMAGE_TOKEN_CACHE_PRELOAD="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD:-1}"
IMAGE_TOKEN_CACHE_READONLY="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY:-1}"
IMAGE_TOKEN_CACHE_WRITE_MISSES="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES:-0}"
IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT:-10}"
IMAGE_TOKEN_MEMORY_CACHE_SIZE="${VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE:-131072}"
IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS:-5000}"
IMAGE_TOKEN_SQLITE_MMAP_SIZE="${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE:-1073741824}"
IMAGE_TOKEN_CACHE_DEBUG="${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG:-0}"
PREFETCH_EMU35_VISION_TOKENIZER=true
ENABLE_WANDB="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lmms-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-rkreft-eth-z-rich}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-eval}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"
WANDB_ARGS="${WANDB_ARGS:-}"
WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_J0zUo6eGaEJKpgoMUbVP1CgsgJ2_CfbfOvYTXIOayJOEpSGp51jtUCUdp162z^CUuA0fytUP}"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# REPO_ROOT="${REPO_ROOT:-${LOCAL_REPO_ROOT}}"
# if [[ ! -d "${REPO_ROOT}/vllm" ]]; then
REPO_ROOT="/workspace"
# fi
VLLM_REPO="${VLLM_REPO:-${REPO_ROOT}/vllm}"
LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-${REPO_ROOT}/lmms-eval}"

usage() {
  cat <<'EOF'
Usage:
  sbatch run_lmms_eval_vllm_eval_local_copy.slurm -- [options]

Options:
  --model-path <path_or_hf_id>           Required model path or HF id.
  --tokenizer-path <path>                Tokenizer path.
  --tasks <csv>                          lmms-eval task list.
  --output-path <path>                   Output directory.
  --log-dir <path>                       Log directory.
  --hf-home <path>                       HF_HOME path.
  --nltk-data <path>                     NLTK_DATA path.
  --xdg-cache-home <path>                XDG_CACHE_HOME path.
  --vllm-cache-root <path>               VLLM_CACHE_ROOT path.
  --models-cache <path>                  LMMS_EVAL_MODELS_CACHE path.
  --batch-size <int>                     Per-rank batch size.
  --num-processes <int>                  Accelerate process count.
  --main-process-port <int>              Accelerate main process port.
  --gpu-memory-utilization <float>       vLLM gpu_memory_utilization.
  --max-model-len <int>                  Max model length (>8192).
  --max-num-batched-tokens <int>         vLLM max_num_batched_tokens.
  --trust-remote-code <True|False>       vLLM trust_remote_code.
  --enable-prefix-caching <true|false>   vLLM prefix caching.
  --enforce-eager <true|false>           vLLM enforce_eager.
  --hf-overrides <json>                  vLLM hf_overrides JSON string.
  --gen-kwargs <kv_csv>                  lmms-eval --gen_kwargs string.
  --seed <seed_spec>                     lmms-eval deterministic seed tuple
                                         (e.g. 0,1234,1234,1234 or single int).
  --limit <int_or_float>                 lmms-eval --limit.
  --chat-template <path_or_template>     chat template for vLLM.
  --extra-model-args <kv_csv>            Extra --model_args suffix.
  --offline-datasets <true|false>        Default: false.
  --enable-image-token-cache <true|false>
                                         Enable Apertus image tokenization cache.
  --image-token-cache-dir <path>         Existing directory for SQLite cache.
  --image-token-cache-collision-guard <0|1|true|false>
                                         Enable collision guard mode.
  --image-token-cache-local-copy <0|1|true|false>
                                         Copy SQLite cache to node-local storage before launch. Default: 1.
  --image-token-cache-local-base <path>  Node-local base directory for copied image-token cache.
                                         Default: /tmp/${USER}/apertus_image_token_cache
  --image-token-cache-preload <0|1|true|false>
                                         Preload SQLite entries into process-local memory cache. Default: 1.
  --image-token-cache-readonly <0|1|true|false>
                                         Open/use image-token cache as read-only during eval. Default: 1.
  --image-token-cache-write-misses <0|1|true|false>
                                         Persist misses to SQLite during eval. Default: 0.
  Environment override:
    VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT
                                         Safety margin (%) for node-local free-space preflight before copying cache. Default: 10.
  --image-token-memory-cache-size <int>  In-process LRU size.
  --image-token-sqlite-busy-timeout-ms <int>
                                         SQLite busy timeout in milliseconds.
  --image-token-sqlite-mmap-size <bytes> SQLite mmap_size hint in bytes.
  --image-token-cache-debug <0|1|true|false>
                                         Enable cache debug logs.
  --prefetch-emu35-vision-tokenizer <true|false>
                                         Prefetch BAAI/Emu3.5-VisionTokenizer once before launch.
  --enable-wandb <true|false>            Enable W&B metric logging. Default: true.
  --wandb-project <name>                 W&B project. Default: lmms-eval.
  --wandb-entity <name>                  W&B entity/team. Default: rkreft-eth-z-rich.
  --wandb-job-type <name>                W&B job_type. Default: eval.
  --wandb-group <name>                   W&B run group.
  --wandb-run-name <name>                W&B run name. Default: model name.
  --wandb-run-id <id>                    W&B run id. Default: deterministic id from model name.
  --wandb-resume <mode>                  W&B resume mode (allow|must|never|auto). Default: allow.
  --wandb-log-samples <true|false>       Also upload eval samples to W&B. Default: false.
  --wandb-args <kv_csv>                  Extra lmms-eval --wandb_args (comma-separated key=value).
  --wandb-api-key <key>                  W&B API key. If omitted, uses WANDB_API_KEY env.
  --debug-mode                           Enable debug mode (sets VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=1).
  --help                                 Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --hf-home) HF_HOME_PATH="$2"; shift 2 ;;
    --nltk-data) NLTK_DATA_PATH="$2"; shift 2 ;;
    --xdg-cache-home) XDG_CACHE_HOME_PATH="$2"; shift 2 ;;
    --vllm-cache-root) VLLM_CACHE_ROOT_PATH="$2"; shift 2 ;;
    --models-cache) LMMS_EVAL_MODELS_CACHE_PATH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --main-process-port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE_CODE="$2"; shift 2 ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="$2"; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER="$2"; shift 2 ;;
    --hf-overrides) HF_OVERRIDES="$2"; shift 2 ;;
    --gen-kwargs) GEN_KWARGS="$2"; shift 2 ;;
    --seed) EVAL_SEED="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
    --extra-model-args) EXTRA_MODEL_ARGS="$2"; shift 2 ;;
    --offline-datasets) OFFLINE_DATASETS="$2"; shift 2 ;;
    --enable-image-token-cache) ENABLE_IMAGE_TOKEN_CACHE="$2"; shift 2 ;;
    --image-token-cache-dir) IMAGE_TOKEN_CACHE_DIR="$2"; shift 2 ;;
    --image-token-cache-collision-guard) IMAGE_TOKEN_CACHE_COLLISION_GUARD="$2"; shift 2 ;;
    --image-token-cache-local-copy) IMAGE_TOKEN_CACHE_LOCAL_COPY="$2"; shift 2 ;;
    --image-token-cache-local-base) IMAGE_TOKEN_CACHE_LOCAL_BASE="$2"; shift 2 ;;
    --image-token-cache-preload) IMAGE_TOKEN_CACHE_PRELOAD="$2"; shift 2 ;;
    --image-token-cache-readonly) IMAGE_TOKEN_CACHE_READONLY="$2"; shift 2 ;;
    --image-token-cache-write-misses) IMAGE_TOKEN_CACHE_WRITE_MISSES="$2"; shift 2 ;;
    --image-token-memory-cache-size) IMAGE_TOKEN_MEMORY_CACHE_SIZE="$2"; shift 2 ;;
    --image-token-sqlite-busy-timeout-ms) IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="$2"; shift 2 ;;
    --image-token-sqlite-mmap-size) IMAGE_TOKEN_SQLITE_MMAP_SIZE="$2"; shift 2 ;;
    --image-token-cache-debug) IMAGE_TOKEN_CACHE_DEBUG="$2"; shift 2 ;;
    --prefetch-emu35-vision-tokenizer) PREFETCH_EMU35_VISION_TOKENIZER="$2"; shift 2 ;;
    --enable-wandb) ENABLE_WANDB="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-job-type) WANDB_JOB_TYPE="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb-run-id) WANDB_RUN_ID="$2"; shift 2 ;;
    --wandb-resume) WANDB_RESUME="$2"; shift 2 ;;
    --wandb-log-samples) WANDB_LOG_SAMPLES="$2"; shift 2 ;;
    --wandb-args) WANDB_ARGS="$2"; shift 2 ;;
    --wandb-api-key) WANDB_API_KEY="$2"; shift 2 ;;
    --debug-mode) IMAGE_TOKEN_CACHE_DEBUG=1; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

is_true() {
  local value
  value="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# Ensure OCR-safe default generation cap.
# If max_new_tokens is absent, append/set it to 32768.
# If caller already set max_new_tokens, keep caller value.
if [[ -z "${GEN_KWARGS}" ]]; then
  GEN_KWARGS="max_new_tokens=32768"
elif [[ ! "${GEN_KWARGS}" =~ (^|,)max_new_tokens= ]]; then
  GEN_KWARGS="${GEN_KWARGS},max_new_tokens=32768"
fi

########################################
# Runtime environment
########################################
export API_TYPE=dummy
export PATH="/opt/venv/bin:${PATH}"
export PYTHONPATH="${LMMS_EVAL_REPO}:${VLLM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME_PATH}"
export NLTK_DATA="${NLTK_DATA_PATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_PATH}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT_PATH}"
export LMMS_EVAL_MODELS_CACHE="${LMMS_EVAL_MODELS_CACHE_PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
if is_true "${ENABLE_WANDB}"; then
  if [[ -z "${WANDB_API_KEY}" && -t 0 ]]; then
    read -r -s -p "Enter WANDB_API_KEY: " WANDB_API_KEY
    echo
  fi
  if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: W&B enabled but API key is missing. Pass --wandb-api-key or set WANDB_API_KEY."
    exit 1
  fi
  export WANDB_API_KEY
  export WANDB_ENTITY="${WANDB_ENTITY}"
  export WANDB_PROJECT="${WANDB_PROJECT}"
fi
# Keep Python hash() stable across processes/runs (affects some internal naming).
HASH_SEED="${EVAL_SEED%%,*}"
if ! [[ "${HASH_SEED}" =~ ^[0-9]+$ ]]; then
  HASH_SEED="${SEED}"
fi
export PYTHONHASHSEED="${PYTHONHASHSEED:-${HASH_SEED}}"
export VLLM_SEED="${VLLM_SEED:-${SEED}}"
export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG

if is_true "${OFFLINE_DATASETS}"; then
  export HF_DATASETS_OFFLINE=1
else
  unset HF_DATASETS_OFFLINE || true
fi

if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  SHARED_IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_DIR}"
  LOCAL_JOB_CACHE_DIR="(not used)"

  echo "IMAGE_TOKEN_CACHE_LOCAL_COPY=${IMAGE_TOKEN_CACHE_LOCAL_COPY}"
  echo "Shared image-token cache dir: ${SHARED_IMAGE_TOKEN_CACHE_DIR}"

  if is_true "${IMAGE_TOKEN_CACHE_LOCAL_COPY}"; then
    LOCAL_JOB_CACHE_DIR="${IMAGE_TOKEN_CACHE_LOCAL_BASE}/${SLURM_JOB_ID:-manual}"
    if ! [[ "${IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT}" =~ ^[0-9]+$ ]]; then
      echo "Warning: invalid VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT=${IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT}; using 10."
      IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT=10
    fi
    mkdir -p "${LOCAL_JOB_CACHE_DIR}"
    echo "Node-local image-token cache dir: ${LOCAL_JOB_CACHE_DIR}"
    du -sh "${SHARED_IMAGE_TOKEN_CACHE_DIR}" || true
    if [[ -d "${SHARED_IMAGE_TOKEN_CACHE_DIR}" ]]; then
      LOCAL_CACHE_CHECK_PATH="${IMAGE_TOKEN_CACHE_LOCAL_BASE}"
      if [[ ! -e "${LOCAL_CACHE_CHECK_PATH}" ]]; then
        LOCAL_CACHE_CHECK_PATH="/tmp"
      fi
      SRC_CACHE_BYTES="$(du -sb "${SHARED_IMAGE_TOKEN_CACHE_DIR}" | awk '{print $1}')"
      LOCAL_CACHE_AVAIL_BYTES="$(df -B1 --output=avail "${LOCAL_CACHE_CHECK_PATH}" | tail -n1 | tr -d '[:space:]')"
      LOCAL_CACHE_REQUIRED_BYTES=$((SRC_CACHE_BYTES + (SRC_CACHE_BYTES * IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT / 100)))
      echo "Local-cache preflight: source_bytes=${SRC_CACHE_BYTES} available_bytes=${LOCAL_CACHE_AVAIL_BYTES} required_bytes=${LOCAL_CACHE_REQUIRED_BYTES} margin_pct=${IMAGE_TOKEN_CACHE_LOCAL_REQUIRED_MARGIN_PCT}"
      if (( LOCAL_CACHE_AVAIL_BYTES < LOCAL_CACHE_REQUIRED_BYTES )); then
        echo "ERROR: insufficient local space at ${LOCAL_CACHE_CHECK_PATH} for cache copy (need ${LOCAL_CACHE_REQUIRED_BYTES} bytes, have ${LOCAL_CACHE_AVAIL_BYTES})."
        exit 1
      fi
      SQLITE_FILES=()
      while IFS= read -r sqlite_file; do
        SQLITE_FILES+=("${sqlite_file}")
      done < <(find "${SHARED_IMAGE_TOKEN_CACHE_DIR}" -type f -name '*.sqlite3*')
      if [[ "${#SQLITE_FILES[@]}" -gt 0 ]]; then
        for sqlite_file in "${SQLITE_FILES[@]}"; do
          rel_path="${sqlite_file#${SHARED_IMAGE_TOKEN_CACHE_DIR}/}"
          rel_dir="$(dirname "${rel_path}")"
          mkdir -p "${LOCAL_JOB_CACHE_DIR}/${rel_dir}"
          cp -f "${sqlite_file}" "${LOCAL_JOB_CACHE_DIR}/${rel_dir}/"
        done
      else
        echo "Warning: no SQLite cache files found in ${SHARED_IMAGE_TOKEN_CACHE_DIR}; local cache starts empty."
      fi
    else
      echo "Warning: shared cache directory ${SHARED_IMAGE_TOKEN_CACHE_DIR} does not exist; local cache starts empty."
    fi
    du -sh "${LOCAL_JOB_CACHE_DIR}" || true
    IMAGE_TOKEN_CACHE_DIR="${LOCAL_JOB_CACHE_DIR}"
  else
    echo "Node-local image-token cache dir: (disabled)"
  fi

  mkdir -p "${IMAGE_TOKEN_CACHE_DIR}"
  echo "Final image-token cache dir: ${IMAGE_TOKEN_CACHE_DIR}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR="${IMAGE_TOKEN_CACHE_DIR}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD="${IMAGE_TOKEN_CACHE_COLLISION_GUARD}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD="${IMAGE_TOKEN_CACHE_PRELOAD}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY="${IMAGE_TOKEN_CACHE_READONLY}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES="${IMAGE_TOKEN_CACHE_WRITE_MISSES}"
  export VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE="${IMAGE_TOKEN_MEMORY_CACHE_SIZE}"
  export VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS="${IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS}"
  export VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE="${IMAGE_TOKEN_SQLITE_MMAP_SIZE}"
  export VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG="${IMAGE_TOKEN_CACHE_DEBUG}"
else
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES || true
  unset VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE || true
  unset VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS || true
  unset VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE || true
  unset VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG || true
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must be set."
  exit 1
fi

if [[ "${MAX_MODEL_LEN}" -le 8192 ]]; then
  echo "MAX_MODEL_LEN must be > 8192 (got ${MAX_MODEL_LEN})."
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
  set +e
  NUM_GPUS_RAW="$(nvidia-smi -L 2>/dev/null | wc -l)"
  NVS_STATUS=$?
  set -e
  if [[ ${NVS_STATUS} -eq 0 && "${NUM_GPUS_RAW}" -gt 0 ]]; then
    NUM_GPUS="${NUM_GPUS_RAW}"
  else
    set +e
    TORCH_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
    TORCH_STATUS=$?
    set -e
    if [[ ${TORCH_STATUS} -eq 0 && "${TORCH_GPU_COUNT}" -gt 0 ]]; then
      NUM_GPUS="${TORCH_GPU_COUNT}"
      echo "Warning: nvidia-smi probe failed (status=${NVS_STATUS}); using torch.cuda.device_count=${NUM_GPUS}."
    else
      NUM_GPUS=0
      echo "Warning: unable to detect GPUs via nvidia-smi or torch; assuming 0 visible GPUs."
    fi
  fi
fi

if [[ "${NUM_GPUS}" -lt "${NUM_PROCESSES}" ]]; then
  echo "Need at least ${NUM_PROCESSES} visible GPUs, found ${NUM_GPUS}."
  exit 1
fi

mkdir -p "${OUTPUT_PATH}" "${NLTK_DATA}" "${LOG_DIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${LMMS_EVAL_MODELS_CACHE}"

# Ensure ANTLR runtime version expected by OmegaConf/Emu3.5 tokenizer path.
# This keeps jobs robust even if the base image has a newer incompatible ANTLR.
if [[ "${PIN_ANTLR_RUNTIME:-true}" == "true" ]]; then
  /opt/venv/bin/pip install --disable-pip-version-check --no-cache-dir --quiet --force-reinstall \
    "antlr4-python3-runtime==4.9.3"
fi

# Fail fast on known ANTLR/OmegaConf mismatch that otherwise appears late.
python - <<'PY'
import importlib.util
import os
import sys

try:
    import omegaconf  # noqa: F401
except Exception as exc:
    print("Preflight failed: cannot import omegaconf.", file=sys.stderr)
    print(f"Root cause: {exc}", file=sys.stderr)
    sys.exit(2)

emu35_root = os.getenv("VLLM_APERTUS_EMU35_CODEBASE", "/workspace/Emu3.5")
module_path = os.path.join(emu35_root, "src", "vision_tokenizer", "__init__.py")
if not os.path.isfile(module_path):
    print(f"Preflight failed: missing Emu3.5 vision tokenizer module: {module_path}", file=sys.stderr)
    sys.exit(2)

spec = importlib.util.spec_from_file_location(
    "emu35_preflight_module",
    module_path,
    submodule_search_locations=[os.path.dirname(module_path)],
)
if spec is None or spec.loader is None:
    print(f"Preflight failed: unable to build import spec for {module_path}", file=sys.stderr)
    sys.exit(2)

module = importlib.util.module_from_spec(spec)
sys.modules["emu35_preflight_module"] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:
    print("Preflight failed: cannot import Emu3.5 vision tokenizer module.", file=sys.stderr)
    print(f"Root cause: {exc}", file=sys.stderr)
    sys.exit(2)
PY

if is_true "${PREFETCH_EMU35_VISION_TOKENIZER}"; then
  PREFETCH_DIR="${LMMS_EVAL_MODELS_CACHE}/BAAI/Emu3.5-VisionTokenizer"
  if [[ ! -f "${PREFETCH_DIR}/config.yaml" || ! -f "${PREFETCH_DIR}/model.ckpt" ]]; then
    echo "Prefetching BAAI/Emu3.5-VisionTokenizer into ${PREFETCH_DIR}"
    PREFETCH_DIR_ENV="${PREFETCH_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/Emu3.5-VisionTokenizer",
    local_dir=os.environ["PREFETCH_DIR_ENV"],
    allow_patterns=["config.yaml", "model.ckpt"],
)
PY
  fi
fi

MODEL_ARGS="model=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},trust_remote_code=${TRUST_REMOTE_CODE},max_model_len=${MAX_MODEL_LEN},hf_overrides=${HF_OVERRIDES}"
MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS},enable_prefix_caching=${ENABLE_PREFIX_CACHING},enforce_eager=${ENFORCE_EAGER}"

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},chat_template=${CHAT_TEMPLATE}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

MODEL_LABEL="$(basename "${MODEL_PATH}")"
if [[ "${MODEL_LABEL}" == "HF" ]]; then
  MODEL_LABEL="$(basename "$(dirname "${MODEL_PATH}")")"
fi
MODEL_LABEL="${MODEL_LABEL// /_}"
if [[ -z "${WANDB_RUN_NAME}" ]]; then
  WANDB_RUN_NAME="${MODEL_LABEL}"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  WANDB_MODEL_HASH="$(printf "%s" "${MODEL_LABEL}" | sha1sum | awk '{print substr($1,1,12)}')"
  WANDB_RUN_ID="model_${WANDB_MODEL_HASH}"
fi

append_wandb_arg() {
  local key="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    return
  fi
  if [[ -n "${FINAL_WANDB_ARGS}" ]]; then
    FINAL_WANDB_ARGS+=","
  fi
  FINAL_WANDB_ARGS+="${key}=${value}"
}

wandb_arg_exists() {
  local key="$1"
  [[ ",${FINAL_WANDB_ARGS}," == *",${key}="* ]]
}

FINAL_WANDB_ARGS="${WANDB_ARGS}"
if ! wandb_arg_exists "project"; then
  append_wandb_arg "project" "${WANDB_PROJECT}"
fi
if ! wandb_arg_exists "entity"; then
  append_wandb_arg "entity" "${WANDB_ENTITY}"
fi
if ! wandb_arg_exists "job_type"; then
  append_wandb_arg "job_type" "${WANDB_JOB_TYPE}"
fi
if ! wandb_arg_exists "group"; then
  append_wandb_arg "group" "${WANDB_GROUP}"
fi
if ! wandb_arg_exists "name"; then
  append_wandb_arg "name" "${WANDB_RUN_NAME}"
fi
if ! wandb_arg_exists "id"; then
  append_wandb_arg "id" "${WANDB_RUN_ID}"
fi
if ! wandb_arg_exists "resume"; then
  append_wandb_arg "resume" "${WANDB_RESUME}"
fi

CMD=(
  accelerate launch
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --main_process_port "${MAIN_PROCESS_PORT}"
  -m lmms_eval
  --model vllm
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --seed "${EVAL_SEED}"
  --output_path "${OUTPUT_PATH}"
  --log_samples
  --log_samples_suffix "vllm_prod"
)

if [[ -n "${GEN_KWARGS}" ]]; then
  CMD+=(--gen_kwargs "${GEN_KWARGS}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if is_true "${ENABLE_WANDB}" && [[ -n "${FINAL_WANDB_ARGS}" ]]; then
  CMD+=(--wandb_args "${FINAL_WANDB_ARGS}")
  if is_true "${WANDB_LOG_SAMPLES}"; then
    CMD+=(--wandb_log_samples)
  fi
fi

/opt/venv/bin/python -c "import nltk; nltk.download('wordnet')"

echo "========== lmms-eval prod launch =========="
echo "VLLM_REPO=${VLLM_REPO}"
echo "LMMS_EVAL_REPO=${LMMS_EVAL_REPO}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "TASKS=${TASKS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "HF_OVERRIDES=${HF_OVERRIDES}"
echo "OFFLINE_DATASETS=${OFFLINE_DATASETS}"
echo "EVAL_SEED=${EVAL_SEED}"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "VLLM_SEED=${VLLM_SEED}"
echo "ENABLE_WANDB=${ENABLE_WANDB}"
if is_true "${ENABLE_WANDB}"; then
  echo "WANDB_ARGS=${FINAL_WANDB_ARGS}"
  echo "WANDB_LOG_SAMPLES=${WANDB_LOG_SAMPLES}"
fi
echo "ENABLE_IMAGE_TOKEN_CACHE=${ENABLE_IMAGE_TOKEN_CACHE}"
if is_true "${ENABLE_IMAGE_TOKEN_CACHE}"; then
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE=${VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS=${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE=${VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_COPY=${IMAGE_TOKEN_CACHE_LOCAL_COPY}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_LOCAL_BASE=${IMAGE_TOKEN_CACHE_LOCAL_BASE}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY}"
  echo "VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES=${VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES}"
fi
echo "HF_HOME=${HF_HOME}"
echo "NLTK_DATA=${NLTK_DATA}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "LMMS_EVAL_MODELS_CACHE=${LMMS_EVAL_MODELS_CACHE}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "PREFETCH_EMU35_VISION_TOKENIZER=${PREFETCH_EMU35_VISION_TOKENIZER}"
if [[ -n "${GEN_KWARGS}" ]]; then
  echo "GEN_KWARGS=${GEN_KWARGS}"
fi
echo "==========================================="

"${CMD[@]}"

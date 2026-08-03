#!/bin/bash
set -euo pipefail

cd /mnt/umm/users/pufanyi/workspace/lmms-eval-vllm

MODEL="/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/models/Wan2.2-I2V-A14B-Diffusers"
TASKS="vbvr"
TP=1
DP=8
GPUS="0,1,2,3,4,5,6,7"
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=1
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=5.0
NUM_FRAMES=81
HEIGHT=384
WIDTH=384
FPS=16
SEED=42
OUTPUT_PATH="/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/eval_out/vbvr_wan22_vllm_omni_external_dp8"
VBVR_ROOT="/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/datasets/VBVR-Bench"
HF_CACHE="/tmp/lmms_eval_hf_vbvr"
LIMIT=""
TQDM_MODE="rank"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --dp) DP="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
    --guidance-scale) GUIDANCE_SCALE="$2"; shift 2 ;;
    --num-frames) NUM_FRAMES="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --vbvr-root) VBVR_ROOT="$2"; shift 2 ;;
    --hf-cache) HF_CACHE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --tqdm-mode) TQDM_MODE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if (( TP > 1 && DP > 1 )); then
  echo "This script supports external DP or internal TP, but not TP>1 and DP>1 together." >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
EXPECTED_GPUS=$(( DP > 1 ? DP : TP ))
if (( ${#GPU_IDS[@]} != EXPECTED_GPUS )); then
  echo "Expected ${EXPECTED_GPUS} GPUs for TP=${TP}, DP=${DP}, got ${#GPU_IDS[@]}: ${GPUS}" >&2
  exit 1
fi

VIDEO_OUTPUT_DIR="${OUTPUT_PATH}/videos"

export HF_HOME="$HF_CACHE"
export VBVR_GT_PATH="$VBVR_ROOT"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000

INTERNAL_TP="$TP"
if (( DP > 1 )); then
  INTERNAL_TP=1
fi

MODEL_ARGS="model=${MODEL},tensor_parallel_size=${INTERNAL_TP},data_parallel_size=1,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
MODEL_ARGS="${MODEL_ARGS},output_dir=${VIDEO_OUTPUT_DIR},output_modalities=video,cache_backend=cache_dit"
MODEL_ARGS="${MODEL_ARGS},num_inference_steps=${NUM_INFERENCE_STEPS},guidance_scale=${GUIDANCE_SCALE},num_frames=${NUM_FRAMES},height=${HEIGHT},width=${WIDTH},fps=${FPS},seed=${SEED}"
MODEL_ARGS="${MODEL_ARGS},tqdm_mode=${TQDM_MODE}"

EVAL_ARGS=(
  eval
  --model vllm_omni
  --model_args "$MODEL_ARGS"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --log_samples
  --output_path "$OUTPUT_PATH"
)

if [[ -n "$LIMIT" ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi

if (( DP > 1 )); then
  CMD=(.venv/bin/torchrun --standalone --nproc_per_node "$DP" -m lmms_eval "${EVAL_ARGS[@]}")
else
  CMD=(.venv/bin/python -m lmms_eval "${EVAL_ARGS[@]}")
fi

echo "Running vLLM-Omni local eval: external DP=${DP}, internal TP=${INTERNAL_TP}, GPUs=${GPUS}, size=${WIDTH}x${HEIGHT}, tqdm=${TQDM_MODE}"
printf '%q ' "${CMD[@]}"
echo ""

"${CMD[@]}"

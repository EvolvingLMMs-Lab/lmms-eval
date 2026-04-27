#!/usr/bin/env bash
set -euo pipefail

cd /mnt/umm/users/pufanyi/workspace/lmms-eval-vllm

MODEL_DIR=${MODEL_DIR:-/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/models/Wan2.2-I2V-A14B-Diffusers}
VBVR_ROOT=${VBVR_ROOT:-/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/datasets/VBVR-Bench}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/eval_out/vbvr_wan22_vllm_omni_local_dp8}

SPLIT=${SPLIT:-all}
LIMIT=${LIMIT:-}

TP=${TP:-1}
DP=${DP:-}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
NUM_FRAMES=${NUM_FRAMES:-81}
HEIGHT=${HEIGHT:-384}
WIDTH=${WIDTH:-384}
FPS=${FPS:-16}
SEED=${SEED:-42}
BOUNDARY_RATIO=${BOUNDARY_RATIO:-}
FLOW_SHIFT=${FLOW_SHIFT:-}

CACHE_BACKEND=${CACHE_BACKEND:-cache_dit}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-}
OVERWRITE=${OVERWRITE:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
RUN_NAME=${RUN_NAME:-}

VISIBLE_GPUS=${VISIBLE_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
WORKERS=${WORKERS:-}
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs}

if [[ -n "$VISIBLE_GPUS" ]]; then
  IFS=',' read -r -a GPU_IDS <<<"$VISIBLE_GPUS"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi

GPU_COUNT=${#GPU_IDS[@]}

if (( GPU_COUNT == 0 )); then
  echo "No visible GPUs found." >&2
  exit 1
fi

if [[ -z "$DP" ]]; then
  DP=$GPU_COUNT
fi

if [[ -z "$DIFFUSION_BATCH_SIZE" ]]; then
  DIFFUSION_BATCH_SIZE=$DP
fi

if [[ -z "$REQUEST_BATCH_SIZE" ]]; then
  REQUEST_BATCH_SIZE=$DP
fi

GPUS_PER_WORKER=$((TP * DP))

if (( GPUS_PER_WORKER < 1 )); then
  echo "Invalid parallelism: TP=$TP DP=$DP" >&2
  exit 1
fi

if [[ -z "$WORKERS" ]]; then
  WORKERS=$((GPU_COUNT / GPUS_PER_WORKER))
fi

if (( WORKERS < 1 )); then
  echo "No worker can be launched with GPU_COUNT=$GPU_COUNT and GPUS_PER_WORKER=$GPUS_PER_WORKER." >&2
  exit 1
fi

MAX_WORKERS=$((GPU_COUNT / GPUS_PER_WORKER))
if (( WORKERS > MAX_WORKERS )); then
  echo "WORKERS=$WORKERS exceeds the available GPU groups ($MAX_WORKERS)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

COMMON_ARGS=(
  --model "$MODEL_DIR"
  --vbvr-root "$VBVR_ROOT"
  --output-root "$OUTPUT_ROOT"
  --split "$SPLIT"
  --tensor-parallel-size "$TP"
  --data-parallel-size "$DP"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --cache-backend "$CACHE_BACKEND"
  --diffusion-batch-size "$DIFFUSION_BATCH_SIZE"
  --request-batch-size "$REQUEST_BATCH_SIZE"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --num-frames "$NUM_FRAMES"
  --height "$HEIGHT"
  --width "$WIDTH"
  --fps "$FPS"
  --seed "$SEED"
)

if [[ -n "$LIMIT" ]]; then
  COMMON_ARGS+=(--limit "$LIMIT")
fi

if [[ -n "$FLOW_SHIFT" ]]; then
  COMMON_ARGS+=(--flow-shift "$FLOW_SHIFT")
fi

if [[ -n "$BOUNDARY_RATIO" ]]; then
  COMMON_ARGS+=(--boundary-ratio "$BOUNDARY_RATIO")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ -n "$RUN_NAME" ]]; then
  COMMON_ARGS+=(--run-name "$RUN_NAME")
fi

PIDS=()

cleanup_children() {
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup_children INT TERM

for ((worker_idx = 0; worker_idx < WORKERS; worker_idx++)); do
  start=$((worker_idx * GPUS_PER_WORKER))
  worker_gpus=("${GPU_IDS[@]:start:GPUS_PER_WORKER}")
  if (( ${#worker_gpus[@]} != GPUS_PER_WORKER )); then
    echo "Worker $worker_idx expected $GPUS_PER_WORKER GPUs but found ${#worker_gpus[@]}." >&2
    cleanup_children
    exit 1
  fi

  worker_visible_gpus=$(IFS=','; echo "${worker_gpus[*]}")
  worker_log="$LOG_DIR/worker_${worker_idx}.log"

  echo "Launching worker $worker_idx/$((WORKERS - 1)) on GPUs [$worker_visible_gpus] -> $worker_log"
  CUDA_VISIBLE_DEVICES="$worker_visible_gpus" \
    .venv/bin/python tools/run_vllm_omni_vbvr_local_parallel.py \
    "${COMMON_ARGS[@]}" \
    --shard-id "$worker_idx" \
    --num-shards "$WORKERS" \
    --skip-eval \
    >"$worker_log" 2>&1 &

  PIDS+=($!)
done

worker_failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    worker_failed=1
  fi
done

if (( worker_failed != 0 )); then
  echo "One or more generation workers failed. Check $LOG_DIR." >&2
  exit 1
fi

if [[ "$SKIP_EVAL" == "1" ]]; then
  exit 0
fi

echo "All generation workers finished. Running final evaluation."
.venv/bin/python tools/run_vllm_omni_vbvr_local_parallel.py \
  "${COMMON_ARGS[@]}" \
  --skip-generate

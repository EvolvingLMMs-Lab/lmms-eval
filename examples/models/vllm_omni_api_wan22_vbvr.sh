#!/bin/bash
set -euo pipefail

cd /mnt/umm/users/pufanyi/workspace/lmms-eval-vllm

MODEL="/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/models/Wan2.2-I2V-A14B-Diffusers"
TASKS="vbvr"
BASE_URL="${BASE_URL:-http://127.0.0.1:8091}"
TP=8
NUM_CPUS=8
OUTPUT_PATH="/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/eval_out/vbvr_wan22_vllm_omni_api_tp8"
VIDEO_OUTPUT_DIR="${OUTPUT_PATH}/videos"

export HF_HOME="${HF_HOME:-/tmp/lmms_eval_hf_vbvr}"
export VBVR_GT_PATH="${VBVR_GT_PATH:-/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/datasets/VBVR-Bench}"

# Start the server separately with TP=8, for example:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "$MODEL" --omni --port 8091 --tensor-parallel-size "$TP"

MODEL_ARGS="base_url=${BASE_URL},output_dir=${VIDEO_OUTPUT_DIR},num_cpus=${NUM_CPUS},timeout=1200"
MODEL_ARGS="${MODEL_ARGS},num_inference_steps=50,guidance_scale=5.0,num_frames=81,height=384,width=384,fps=16,seed=42"

CMD=".venv/bin/python -m lmms_eval eval \
    --model vllm_omni_api \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --output_path ${OUTPUT_PATH}"

echo "Running command:"
echo "$CMD"
echo ""

eval "$CMD"

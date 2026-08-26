#!/usr/bin/env bash
set -euo pipefail

# SGLang Diffusion currently requires a newer PyAV than lmms-eval's base
# environment. A dedicated uv environment keeps the two dependency sets apart:
#   uv venv .venv-sglang --python 3.12
#   uv pip install --python .venv-sglang/bin/python -e .
#   uv pip install --python .venv-sglang/bin/python --prerelease=allow "sglang[diffusion]==0.5.10.post1"
#   SGLANG_PYTHON=.venv-sglang/bin/python bash examples/models/sglang_wan22_vbvr.sh
#
# DiffGenerator owns its worker processes, so run one lmms-eval process and
# configure intra-sample parallelism with NUM_GPUS rather than accelerate.

MODEL_PATH=${MODEL_PATH:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}
TASKS=${TASKS:-vbvr}
NUM_GPUS=${NUM_GPUS:-4}
SGLANG_PYTHON=${SGLANG_PYTHON:-.venv/bin/python}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/sglang_wan22_vbvr}
VIDEO_DIR=${VIDEO_DIR:-${OUTPUT_PATH}/videos}
LIMIT=${LIMIT:-}
DIT_LAYERWISE_OFFLOAD=${DIT_LAYERWISE_OFFLOAD:-}
DIT_CPU_OFFLOAD=${DIT_CPU_OFFLOAD:-}

PARALLEL_ARGS="num_gpus=${NUM_GPUS}"
if (( NUM_GPUS >= 2 && NUM_GPUS % 2 == 0 )); then
    PARALLEL_ARGS="${PARALLEL_ARGS},enable_cfg_parallel=true,ulysses_degree=$((NUM_GPUS / 2))"
fi

MODEL_ARGS="model=${MODEL_PATH},runtime=diffusion,output_dir=${VIDEO_DIR}"
MODEL_ARGS="${MODEL_ARGS},${PARALLEL_ARGS},text_encoder_cpu_offload=true,pin_cpu_memory=true"
if [[ -n "${DIT_LAYERWISE_OFFLOAD}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},dit_layerwise_offload=${DIT_LAYERWISE_OFFLOAD}"
fi
if [[ -n "${DIT_CPU_OFFLOAD}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},dit_cpu_offload=${DIT_CPU_OFFLOAD}"
fi
MODEL_ARGS="${MODEL_ARGS},num_frames=81,height=480,width=832,num_inference_steps=40"
MODEL_ARGS="${MODEL_ARGS},guidance_scale=3.5,guidance_scale_2=3.5,fps=16,seed=42"

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi

"${SGLANG_PYTHON}" -m lmms_eval \
    --model sglang \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix sglang_wan22 \
    --output_path "${OUTPUT_PATH}" \
    "${EXTRA_ARGS[@]}"

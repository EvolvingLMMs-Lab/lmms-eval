#!/usr/bin/env bash
# Frames-backend launcher (uniform sampling) for LLaVA-OneVision-2-8B-Instruct.
# Usage: TASK=<task> F=<num_frames> MP=<max_pixels> bash run_frames.sh
# Optional: PORT, NPROC, MODEL, OUT_ROOT, IL=1 (sets LMMS_IL_NOPREFIX/FILTER_NOISE).
set -euo pipefail
: "${TASK:?TASK is required}"
: "${F:?F (max_num_frames) is required}"
: "${MP:?MP (min_pixels = max_pixels) is required}"
PORT=${PORT:-29830}
NPROC=${NPROC:-8}

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MODEL:-lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct}
OUT_ROOT=${OUT_ROOT:-${REPO}/out}
OUT_DIR=${OUT_ROOT}/${TASK}_frames_f${F}_mp${MP}
mkdir -p "${OUT_DIR}"

export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
if [[ "${IL:-0}" == "1" ]]; then
  export LMMS_IL_NOPREFIX=1
  export LMMS_IL_FILTER_NOISE=1
fi

cd "${REPO}"
accelerate launch --num_processes=${NPROC} --main_process_port=${PORT} -m lmms_eval \
  --model llava_onevision2 \
  --model_args "pretrained=${MODEL},trust_remote_code=True,attn_implementation=flash_attention_2,messages_format=timestamp,fps=1,max_num_frames=${F},min_pixels=${MP},max_pixels=${MP},video_backend=frames" \
  --tasks ${TASK} \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUT_DIR}/" \
  2>&1 | tee "${OUT_DIR}/run.log"

#!/usr/bin/env bash
# Codec-backend launcher (canvas-packed video tokens) for LLaVA-OneVision-2-8B-Instruct.
# Usage: TASK=<task> TC=<num_canvases> [TS=<timestamp_decimals>] [IL=1] bash run_codec.sh
# Optional: PORT, NPROC, MODEL, OUT_ROOT, MIN_PX (default 100352), MAX_PX (default 313600),
#           LLAVA_CODEC_OFFLINE_ROOT (offline assets) or ONLINE_CODEC_CACHE_DIR (online cache).
set -euo pipefail
: "${TASK:?TASK is required}"
: "${TC:?TC (codec_target_canvas / max_num_frames) is required}"
PORT=${PORT:-29830}
NPROC=${NPROC:-8}
TS=${TS:-1}
MIN_PX=${MIN_PX:-100352}
MAX_PX=${MAX_PX:-313600}

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MODEL:-lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct}
OUT_ROOT=${OUT_ROOT:-${REPO}/out}
OUT_DIR=${OUT_ROOT}/${TASK}_codec_tc${TC}_mp${MAX_PX}_ts${TS}
mkdir -p "${OUT_DIR}"

export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export LLAVA_CODEC_ONLINE_TUNED=${LLAVA_CODEC_ONLINE_TUNED:-1}
if [[ "${IL:-0}" == "1" ]]; then
  export LMMS_IL_NOPREFIX=1
  export LMMS_IL_FILTER_NOISE=1
fi
# Online canvas cache (skip re-decoding between runs); unset to disable.
export ONLINE_CODEC_CACHE_DIR=${ONLINE_CODEC_CACHE_DIR:-${OUT_ROOT}/online_codec_${TASK}_tc${TC}}

cd "${REPO}"
accelerate launch --num_processes=${NPROC} --main_process_port=${PORT} -m lmms_eval \
  --model llava_onevision2 \
  --model_args "pretrained=${MODEL},trust_remote_code=True,attn_implementation=flash_attention_2,messages_format=timestamp,timestamp_decimals=${TS},fps=1,max_num_frames=${TC},min_pixels=${MIN_PX},max_pixels=${MAX_PX},video_backend=codec,codec_target_canvas=${TC}" \
  --tasks ${TASK} \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUT_DIR}/" \
  2>&1 | tee "${OUT_DIR}/run.log"

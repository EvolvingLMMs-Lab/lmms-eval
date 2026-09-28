#!/usr/bin/env bash
# Analytical FLOPs per sample (vision encoder / LLM prefill / LLM decode) for finished runs -> <run dir>/flops.json.
# run_eval.sh calls this after every run; use it directly to (re)compute older runs. CPU only, offline.
#
# Usage: jetson/scripts/compute_flops.sh <run dir> [<run dir> ...]
#   jetson/scripts/compute_flops.sh jetson/results/thor/Qwen2.5-VL-3B-Instruct/mme/hf-bf16/20260923-215339
#   jetson/scripts/compute_flops.sh $(ls -d jetson/results/thor/*/*/*/*/)     # every run
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${FLOPS_IMAGE:-lmms-eval-jetson:latest}
source "$REPO/jetson/frameworks/common.sh"
[ $# -ge 1 ] || { echo "usage: $0 <run dir> [...]" >&2; exit 1; }

RUN_DIRS=()
for d in "$@"; do RUN_DIRS+=("$(realpath --relative-to="$REPO" "$d")"); done

docker run --rm --user "$(id -u):$(id -g)" $(shared_group_args) \
  -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -e HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub" \
  -e HF_HUB_OFFLINE="$OFFLINE" -e HF_DATASETS_OFFLINE="$OFFLINE" -e TRANSFORMERS_OFFLINE="$OFFLINE" -e CUDA_VISIBLE_DEVICES= \
  -v "$REPO":"$REPO" -w "$REPO" -e PYTHONPATH="$REPO" -v "$HF_CACHE":"$HF_CACHE" \
  "$IMAGE" python "jetson/models/$MODEL/flops.py" "${RUN_DIRS[@]}"

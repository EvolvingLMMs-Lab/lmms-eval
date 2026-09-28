#!/usr/bin/env bash
# Quantize (optional) and export Qwen2.5-VL to ONNX for TensorRT Edge-LLM (>= 0.8 checkpoint-based workflow).
# Runs on Jetson Thor (tested: L4T r38.4, Edge-LLM v0.10.1, nvcr.io/nvidia/pytorch:26.05-py3) or an x86 GPU host.
# Not on Orin with JetPack 6 (Edge-LLM 0.6 used a different, removed export CLI).
#
# Usage: export.sh <3b|7b> <fp16|int4_awq|fp8|nvfp4> [workspace]
#   workspace default: $TRT_WORKSPACE, else /opt/models/trt-edgellm -> <workspace>/<model>-<precision>/onnx/{llm,visual}
#   fp16      the Hugging Face checkpoint as is
#   int4_awq  tensorrt-edgellm-quantize (AWQ, default text calibration) on the Hugging Face checkpoint.
#             AWQ_SOURCE=qwen exports Qwen's AWQ checkpoint instead ($AWQ_REPO, as used by the vLLM awq runs);
#             with Edge-LLM v0.10.1 on Thor that engine answers text prompts but returns an empty answer for every
#             image prompt, so it is not the default.
#   The vision encoder is always exported in FP16 from the Hugging Face checkpoint. (The AWQ repos' config.json
#   omits the vision fields that have defaults, e.g. num_heads, which Edge-LLM's visual exporter requires; their
#   vision weights are bit-identical to the base checkpoint's.)
#   fp8/nvfp4 tensorrt-edgellm-quantize on the Hugging Face checkpoint (default text calibration, LM head and
#             vision encoder unquantized), then export; fp8/nvfp4 engines run only on Thor (SM110) and Blackwell.
# Checkpoints come from $HF_CACHE (download them first: jetson/scripts/download_assets.sh hf <size> / vllm <size>-awq).
# The quantized checkpoint is deleted after a successful export unless KEEP_QUANTIZED=1.
# Env: EDGELLM_REF (v0.10.1), BASE_IMAGE (nvcr.io/nvidia/pytorch:26.05-py3), EXPORT_IMAGE (built if missing),
#      GPU_FLAGS (default --runtime nvidia; x86 hosts: --gpus all), HF_CACHE, SHARED_GROUP.
set -euo pipefail

SIZE=${1:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4> [workspace]}
PRECISION=${2:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4> [workspace]}
WORKSPACE=$(realpath -m "${3:-${TRT_WORKSPACE:-/opt/models/trt-edgellm}}")
EDGELLM_REF=${EDGELLM_REF:-v0.10.1}
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}
EXPORT_IMAGE=${EXPORT_IMAGE:-lmms-eval-trt-export:$EDGELLM_REF}
GPU_FLAGS=${GPU_FLAGS:---runtime nvidia}

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OFFLINE=1
source "$REPO/jetson/frameworks/common.sh"

load_model "$SIZE" || exit 1
SRC=$(hf_snapshot "$HF_REPO")
case "$PRECISION" in
  fp16|fp8|nvfp4) LLM_CKPT=$SRC ;;
  int4_awq) [ "${AWQ_SOURCE:-quantize}" = qwen ] && LLM_CKPT=$(hf_snapshot "$AWQ_REPO") || LLM_CKPT=$SRC ;;
  *) echo "precision must be fp16, int4_awq, fp8 or nvfp4" >&2; exit 1 ;;
esac
for d in "$SRC" "$LLM_CKPT"; do
  [ -d "$d" ] || { echo "checkpoint $d not in $HF_CACHE - run jetson/scripts/download_assets.sh first" >&2; exit 1; }
done

if ! docker image inspect "$EXPORT_IMAGE" >/dev/null 2>&1; then
  docker build --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg EDGELLM_REF="$EDGELLM_REF" \
    -f "$JETSON/frameworks/trt_edgellm/Dockerfile.export" -t "$EXPORT_IMAGE" "$JETSON/frameworks/trt_edgellm"
fi

OUT=$WORKSPACE/$MODEL_TAG-$PRECISION
mkdir -p "$OUT"
# Calibration datasets (fp8/nvfp4) are downloaded into the shared HF cache, so the Hub is reachable here.
# They are public: HF_TOKEN_PATH points away from a token file in the shared cache that may belong to someone else.
docker run --rm $GPU_FLAGS --ipc=host -e HF_TOKEN_PATH=/tmp/no-hf-token \
  --user "$(id -u):$(id -g)" $(shared_group_args) -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -v "$HF_CACHE":"$HF_CACHE" -v "$WORKSPACE":"$WORKSPACE" \
  -e SRC="$SRC" -e LLM_CKPT="$LLM_CKPT" -e AWQ_SOURCE="${AWQ_SOURCE:-quantize}" -e OUT="$OUT" -e PRECISION="$PRECISION" -e KEEP_QUANTIZED="${KEEP_QUANTIZED:-0}" \
  "$EXPORT_IMAGE" bash -euo pipefail -c '
    umask 002
    if [ "$PRECISION" = fp8 ] || [ "$PRECISION" = nvfp4 ] || { [ "$PRECISION" = int4_awq ] && [ "$AWQ_SOURCE" != qwen ]; }; then
      rm -rf "$OUT/quantized"
      tensorrt-edgellm-quantize llm --model_dir "$SRC" --output_dir "$OUT/quantized" --quantization "$PRECISION"
      LLM_CKPT=$OUT/quantized
    fi
    rm -rf "$OUT/onnx"
    tensorrt-edgellm-export "$LLM_CKPT" "$OUT/onnx" --skip-visual
    tensorrt-edgellm-export "$SRC" "$OUT/onnx" --skip-llm
    [ "$KEEP_QUANTIZED" = 1 ] || rm -rf "$OUT/quantized"
    ls "$OUT/onnx"
  '
echo "exported: $OUT/onnx - next: jetson/frameworks/trt_edgellm/build_engines.sh $SIZE $PRECISION"

#!/usr/bin/env bash
# Build TensorRT Edge-LLM engines on the Jetson from ONNX produced by export.sh (tested on Thor, Edge-LLM v0.10.1).
#
# Usage: build_engines.sh <3b|7b> <fp16|int4_awq|fp8|nvfp4>   (fp8/nvfp4: Thor only)
# Expects  $TRT_WORKSPACE/<model>-<precision>/onnx/{llm,visual}   (TRT_WORKSPACE default /opt/models/trt-edgellm)
# Writes   $TRT_WORKSPACE/<model>-<precision>/engines/{llm,visual}
# REMOVE_ONNX=1 deletes <model>-<precision>/onnx after a successful build (saves disk; export.sh recreates it).
#
# Image-token limits come from the model file (IMAGE_MIN/MAX_TOKENS, 256..2048 for Qwen2.5-VL), so every
# framework sees the same image resolution range.
set -euo pipefail

SIZE=${1:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4>}
PRECISION=${2:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4>}
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}
IMAGE=${IMAGE:-lmms-eval-trt-edgellm:latest}

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$REPO/jetson/frameworks/common.sh"
load_model "$SIZE" || exit 1
DIR=$TRT_WORKSPACE/$MODEL_TAG-$PRECISION
[ -d "$DIR/onnx/llm" ] && [ -d "$DIR/onnx/visual" ] || { echo "missing $DIR/onnx/{llm,visual} - run export.sh on an x86 GPU host or Thor first" >&2; exit 1; }

docker run --rm --runtime nvidia --ipc=host \
  --user "$(id -u):$(id -g)" $(shared_group_args) -e HOME=/tmp \
  -v "$DIR":"$DIR" "$IMAGE" bash -euo pipefail -c "
    umask 002
    llm_build --onnxDir '$DIR/onnx/llm' --engineDir '$DIR/engines/llm' \
      --maxBatchSize 1 --maxInputLen 2560 --maxKVCacheCapacity 3072
    visual_build --onnxDir '$DIR/onnx/visual' --engineDir '$DIR/engines/visual' \
      --minImageTokens $IMAGE_MIN_TOKENS --maxImageTokens $IMAGE_MAX_TOKENS --maxImageTokensPerImage $IMAGE_MAX_TOKENS
  "
echo "engines: $DIR/engines/{llm,visual}"
[ "${REMOVE_ONNX:-0}" = 1 ] && rm -rf "$DIR/onnx" && echo "removed $DIR/onnx"

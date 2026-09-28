# Qwen2.5-VL-Instruct: everything model-specific the harness needs (sourced by frameworks/common.sh).
#
# model_resolve <size> sets:
#   MODEL_TAG     result-dir name and Hub model name, e.g. Qwen2.5-VL-3B-Instruct
#   HF_REPO       base checkpoint (hf, vllm bf16, Edge-LLM export)
#   AWQ_REPO      official AWQ 4-bit checkpoint (vllm awq)
#   GGUF_REPO     GGUF weights + vision projector (llamacpp): <MODEL_TAG>-<Q>.gguf, mmproj-<MODEL_TAG>-f16.gguf
#   HF_BACKEND    lmms-eval model name for the hf framework
# and the image range shared by every framework: 256..2048 visual tokens of 28x28 pixels
# (IMAGE_MIN_PIXELS / IMAGE_MAX_PIXELS, IMAGE_MIN_TOKENS / IMAGE_MAX_TOKENS), the lmms-eval defaults for Qwen2.5-VL.
# FLOPs per sample: models/qwen2_5_vl/flops.py.

MODEL_SIZES="3b 7b"
IMAGE_MIN_TOKENS=256
IMAGE_MAX_TOKENS=2048
IMAGE_MIN_PIXELS=$((IMAGE_MIN_TOKENS * 28 * 28))
IMAGE_MAX_PIXELS=$((IMAGE_MAX_TOKENS * 28 * 28))
HF_BACKEND=qwen2_5_vl

model_resolve() {
  case "${1,,}" in
    3b) MODEL_TAG=Qwen2.5-VL-3B-Instruct ;;
    7b) MODEL_TAG=Qwen2.5-VL-7B-Instruct ;;
    *) echo "unknown model size '$1' (qwen2_5_vl: $MODEL_SIZES)" >&2; return 1 ;;
  esac
  HF_REPO=Qwen/$MODEL_TAG
  AWQ_REPO=Qwen/$MODEL_TAG-AWQ
  GGUF_REPO=ggml-org/$MODEL_TAG-GGUF
}

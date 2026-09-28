#!/usr/bin/env bash
# Follow-up experiments after the first framework comparison (Qwen2.5-VL-3B, Jetson AGX Orin 32GB).
#   E1  image transport overhead: vLLM gets PIL images directly (no PNG/base64 round trip),
#       llama.cpp gets fast lossless PNG (compress level 1) - compare wall time with the default runs
#   E2  decode-heavy task: coco2017_cap_val_lite (500 images, one-sentence captions, up to 64 tokens)
#   E3  image resolution sweep on vLLM bf16: max 256 / 512 / 1024 visual tokens (2048 = E1 run)
# Runs one after another; a failed run is logged and the queue continues.
# Usage: jetson/experiments/qwen2_5_vl/followups.sh
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$REPO/jetson/frameworks/common.sh"  # RESULTS_DIR
RUN=$REPO/jetson/scripts/run_eval.sh
mkdir -p "$REPO/$RESULTS_DIR/logs"
LOG=$REPO/$RESULTS_DIR/logs/followups.log

run() {
  echo "=== $(date +%H:%M:%S) $*" | tee -a "$LOG"
  if env "$@" >/dev/null 2>&1; then echo "    ok" | tee -a "$LOG"; else echo "    FAILED" | tee -a "$LOG"; return 1; fi
}

PIL="EXTRA_MODEL_ARGS=pass_pil_images=True RUN_TAG=pil"
PNG1="EVAL_ENV=LMMS_IMAGE_PNG_COMPRESS_LEVEL=1 RUN_TAG=png1"
pixels() { echo "pass_pil_images=True,max_pixels=$1,mm_processor_kwargs={\"min_pixels\":$(($1 < 200704 ? $1 : 200704)),\"max_pixels\":$1}"; }

# E1 - same MME runs as the baseline, only the image transport differs
run $PIL $RUN vllm 3b mme
run $PNG1 $RUN llamacpp 3b-q8_0 mme

# E2 - captioning (decode-heavy); smoke test first
COCO=coco2017_cap_val_lite
if run $RUN hf 3b $COCO 8; then
  run $RUN hf 3b $COCO
  run $PIL $RUN vllm 3b $COCO
  run $PIL $RUN vllm 3b-awq $COCO
  run $PNG1 $RUN llamacpp 3b-q8_0 $COCO
  run $PNG1 $RUN llamacpp 3b-q4_k_m $COCO
else
  echo "    skipping E2: smoke test failed" | tee -a "$LOG"
fi

# E3 - resolution sweep (1 visual token = 28x28 pixels)
for tokens in 256 512 1024; do
  run "EXTRA_MODEL_ARGS=$(pixels $((tokens * 784)))" RUN_TAG=pil-max${tokens}tok $RUN vllm 3b mme
done

python3 "$REPO/jetson/scripts/summarize.py" >/dev/null
echo "=== $(date +%H:%M:%S) done" | tee -a "$LOG"

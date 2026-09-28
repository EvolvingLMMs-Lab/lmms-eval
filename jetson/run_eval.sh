#!/usr/bin/env bash
# Run one lmms-eval benchmark on Jetson with a chosen inference framework, model size and precision.
#
# Usage: jetson/run_eval.sh <framework> <size>[-<precision>] [tasks=mme] [limit]
#   framework  hf | vllm | llamacpp | trt_edgellm   (definitions in jetson/frameworks/<framework>.sh)
#   size       3b | 7b   (Qwen2.5-VL-3B/7B-Instruct)
#   precision  hf: bf16 | vllm: bf16, awq | llamacpp: q4_k_m, q8_0, f16 | trt_edgellm: fp16, int4_awq
#              (omitted -> first one listed)
# Examples:
#   jetson/run_eval.sh hf 3b mme 8              # smoke test, 8 samples
#   jetson/run_eval.sh vllm 7b-awq mme
#   jetson/run_eval.sh llamacpp 3b-q4_k_m mme
# Env overrides: OFFLINE (default 1: only what download_assets.sh cached), EXTRA_MODEL_ARGS, IMAGE,
#                HF_CACHE (default /opt/hf-cache), plus per-framework ones (see jetson/frameworks/*.sh).
#   RUN_TAG   label for a variant, appended to the result dir: <framework>-<precision>+<RUN_TAG>
#   RESULTS_DIR  results root relative to the repo (default jetson/results; e.g. jetson/results-thor)
#   WAIT_GPU_IDLE  default 1: before starting, wait until no other process uses the GPU (checked with
#             nvidia-smi, e.g. another user's host job that `docker ps` cannot see); 0 = start anyway
#   FLOPS     default 1: after the run, write analytical FLOPs per sample to flops.json (jetson/compute_flops.sh)
#   EVAL_ENV  space-separated KEY=VALUE pairs passed into the eval container
#             (e.g. EVAL_ENV="LMMS_IMAGE_PNG_COMPRESS_LEVEL=1")
#
# Each run writes jetson/results/<model>/<tasks>/<framework>-<precision>[+<tag>]/<timestamp>[_limitN]/:
#   run_info.txt, run.log, tegrastats.log.gz, flops.json (+ flops_samples.jsonl.gz), [server.log | trt_profile.json],
#   lmms_eval/<date>_results.json and lmms_eval/<date>_samples_<task>.jsonl.gz
set -uo pipefail

FRAMEWORK=${1:?usage: $0 <hf|vllm|llamacpp|trt_edgellm> <3b|7b>[-precision] [tasks] [limit]}
MODEL_SPEC=${2:?usage: $0 <framework> <3b|7b>[-precision] [tasks] [limit]}
TASKS=${3:-mme}
LIMIT=${4:-}

REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/jetson/frameworks/common.sh"
resolve_framework "$FRAMEWORK" "$MODEL_SPEC" || exit 1

IMAGE=${IMAGE:-lmms-eval-jetson:latest}
RUN_ID=$(date +%Y%m%d-%H%M%S)${LIMIT:+_limit$LIMIT}
OUT_REL=${RESULTS_DIR:-jetson/results}/$MODEL_TAG/${TASKS//,/+}/$FRAMEWORK-$PRECISION${RUN_TAG:++$RUN_TAG}/$RUN_ID
OUT=$REPO/$OUT_REL
mkdir -p "$OUT"

DOCKER_ARGS=()
for kv in ${EVAL_ENV:-}; do DOCKER_ARGS+=(-e "$kv"); done
fw_setup || exit 1
MODEL_ARGS+=${EXTRA_MODEL_ARGS:+,$EXTRA_MODEL_ARGS}
EVAL_CMD=(python -m lmms_eval --model "$BACKEND" --model_args "$MODEL_ARGS" --tasks "$TASKS"
          --batch_size 1 --log_samples --output_path "$OUT_REL" ${LIMIT:+--limit "$LIMIT"})

# Other GPU compute processes (host jobs included) would skew latency and memory; wait for them.
other_gpu_procs() {
  command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/  */ /g'
}
if [ "${WAIT_GPU_IDLE:-1}" = 1 ] && [ -n "$(other_gpu_procs)" ]; then
  echo "$(date +%H:%M:%S) waiting for other GPU processes to finish: $(other_gpu_procs | tr '\n' ';')"
  while [ -n "$(other_gpu_procs)" ]; do sleep 60; done
  echo "$(date +%H:%M:%S) GPU idle, starting"
fi

{
  echo "date:        $(date -Is)"
  echo "board:       $(tr -d '\0' </proc/device-tree/model)"
  echo "l4t:         $(head -1 /etc/nv_tegra_release)"
  echo "power mode:  $(nvpmodel -q 2>/dev/null | head -1)"
  echo "git commit:  $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' (dirty)')"
  echo "framework:   $FRAMEWORK"
  echo "precision:   $PRECISION"
  [ -n "${RUN_TAG:-}" ] && echo "run tag:     $RUN_TAG"
  [ -n "${EVAL_ENV:-}" ] && echo "eval env:    $EVAL_ENV"
  echo "image:       $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19))"
  [ -n "${LLAMACPP_IMAGE:-}" ] && echo "server image: $LLAMACPP_IMAGE ($(docker image inspect -f '{{.Id}}' "$LLAMACPP_IMAGE" 2>/dev/null | cut -c1-19))"
  echo "other containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  echo "other GPU processes: $(other_gpu_procs | tr '\n' ';')"
  echo "command:     ${EVAL_CMD[*]}"
} >"$OUT/run_info.txt"

tegrastats --interval 1000 --logfile "$OUT/tegrastats.log" &
TEGRA_PID=$!
cleanup() {
  kill "$TEGRA_PID" 2>/dev/null || true
  declare -F fw_stop >/dev/null && fw_stop
}
trap cleanup EXIT

if declare -F fw_start >/dev/null; then
  fw_start || exit 1
fi

# Run as the calling user (+ SHARED_GROUP if it exists) so files in the shared HF cache stay group-writable.
docker run --rm --runtime nvidia --ipc=host "${DOCKER_ARGS[@]}" \
  --user "$(id -u):$(id -g)" $(shared_group_args) \
  -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -e HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub" \
  -e TRANSFORMERS_CACHE="$HF_CACHE/hub" -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_HUB_OFFLINE="$OFFLINE" -e HF_DATASETS_OFFLINE="$OFFLINE" -e TRANSFORMERS_OFFLINE="$OFFLINE" \
  -v "$REPO":"$REPO" -w "$REPO" -e PYTHONPATH="$REPO" -v "$HF_CACHE":"$HF_CACHE" \
  "$IMAGE" bash -c 'umask 002; python -c "import torch,transformers;print(\"torch\",torch.__version__,\"transformers\",transformers.__version__)"; exec "$@"' _ "${EVAL_CMD[@]}" \
  2>&1 | tee "$OUT/run.log"
STATUS=${PIPESTATUS[0]}
# lmms-eval logs evaluation errors but still exits 0.
if [ "$STATUS" -eq 0 ] && grep -q "Error during evaluation" "$OUT/run.log"; then STATUS=1; fi

# lmms-eval names its output dir after the model (a cache path gives "snapshots__<hash>"); use one fixed name.
# Backends without a model path (e.g. trt_edgellm) get no subdir: the files land in $OUT itself.
for d in "$OUT"/*/; do
  ls "$d"*_results.json >/dev/null 2>&1 && [ "$d" != "$OUT/lmms_eval/" ] && mv "$d" "$OUT/lmms_eval"
done
if ls "$OUT"/*_results.json >/dev/null 2>&1; then
  mkdir -p "$OUT/lmms_eval" && mv "$OUT"/*_results.json "$OUT"/*_samples_*.jsonl "$OUT/lmms_eval/" 2>/dev/null
fi

grep -m1 '^torch ' "$OUT/run.log" | sed 's/^/versions:    /' >>"$OUT/run_info.txt" || true
# Analytical FLOPs per sample -> $OUT/flops.json (CPU; a failure here does not fail the run).
if [ "$STATUS" -eq 0 ] && [ "${FLOPS:-1}" = 1 ]; then
  "$REPO/jetson/compute_flops.sh" "$OUT" >"$OUT/flops.log" 2>&1 || echo "FLOPs computation failed, see $OUT/flops.log" >&2
fi
# Shrink for git: strip progress-bar redraws from logs, gzip tegrastats and per-sample files (jetson/slim_run.py).
python3 "$REPO/jetson/slim_run.py" "$OUT" || true
echo "results in: $OUT (exit $STATUS)"
exit "$STATUS"

#!/usr/bin/env bash
# eval.sh — Apertus VLM eval CLI (per-task SQLite cache, single user entry point).
#
# Usage:
#   bash eval.sh <model> [--tasks T | --suite full|smoke] [--mode fill|readonly] [--help]
#
# <model> forms:
#   /path/to/ckpt                       single path
#   /a,/b,/c                            comma-separated paths
#   @file.txt                           one path per line (comments # and blanks OK)
#
# --tasks   comma-separated, or @file.txt. If omitted, falls back to --suite.
# --suite   named curation: full (default, ~46 tasks) | smoke (3-task canary).
# --mode    fill (populate per-task cache) | readonly (default, eval against cache).
#
# Examples:
#   bash eval.sh /path/to/ckpt                          # full suite, readonly
#   bash eval.sh /path/to/ckpt --suite smoke            # 3-task sanity
#   bash eval.sh /path/to/ckpt --tasks mmmu_val,chartqa # specific tasks
#   bash eval.sh /a,/b,/c                               # multiple models
#   bash eval.sh @models.txt --tasks @custom.txt
#   bash eval.sh /path/to/ckpt --mode fill              # populate cache first
#
# Cache layout (per-task SQLite, bounded growth per file):
#   $CACHE_BASE/{task}/image_tokens/apertus_image_token_cache.sqlite3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-${SCRIPT_DIR}/eval_job.slurm}"
CACHE_BASE="${CACHE_BASE:-/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/image_token_cache}"

# ------------------------------------------------------------------
# Canonical task suites (curations — edit here, not in external files)
# ------------------------------------------------------------------
# Smoke: canary subset, ~5 min/model.
SUITE_SMOKE="gqa,mmstar,pope"

# Full: 46-task working set. Excludes:
#   - canary (gqa, mmstar, pope) — run separately via --suite smoke
#   - broken upstream / known-fail: chartqapro, vending_bench2, amber_g, dude,
#     dynamath_reasoning, mme_realworld, mmlongbench_doc, refspatial,
#     seedbench_2, vcr_wiki_en_easy, vcr_wiki_en_hard, viewspatial,
#     where2place, zerobench
SUITE_FULL="realworldqa,seedbench,ocrbench,ocrbench_v2,textvqa_val,docvqa_val,vqav2_val,infovqa_val,chartqa,mme,ai2d,mmmu_val,mathvision_test,mathvision_testmini,VisualPuzzles_direct,countbench,embspatial,screenspot,osworld_g,refcoco,refcoco+,refcocog,mmmu_pro,mathvision_reason_test,mathvision_reason_testmini,blink,cv_bench,mmsi_bench,3dsrbench,site_bench_image,mindcube_tiny,mmvp,cmmmu_val,vlmsareblind,vlms_are_biased,muirbench,erqa,scienceqa,iconqa_val,simplevqa,omnidocbench,visulogic,vstar_bench,screenspot_v2,screenspot_pro,seedbench_2_plus"

# ------------------------------------------------------------------
# CLI parsing
# ------------------------------------------------------------------
usage() {
  awk 'NR>1 && /^#/{sub(/^# ?/, ""); print; next} NR>1 && !/^#/{exit}' "${BASH_SOURCE[0]}"
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

MODELS_RAW=""
TASKS_RAW=""
SUITE=""
MODE="readonly"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  usage; exit 0 ;;
    --tasks)    TASKS_RAW="$2"; shift 2 ;;
    --suite)    SUITE="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --*)        echo "unknown flag: $1" >&2; usage; exit 1 ;;
    *)
      # First positional = model(s)
      if [[ -z "$MODELS_RAW" ]]; then MODELS_RAW="$1"; shift
      else echo "unexpected positional: $1" >&2; usage; exit 1
      fi
      ;;
  esac
done

if [[ -z "$MODELS_RAW" ]]; then echo "missing <model> argument" >&2; usage; exit 1; fi

case "$MODE" in fill|readonly) ;; *) echo "--mode must be fill|readonly (got: $MODE)" >&2; exit 1 ;; esac

# ------------------------------------------------------------------
# Resolve models: inline path, comma-separated, or @file
# ------------------------------------------------------------------
resolve_list() {
  # $1 = raw value ("/path", "/a,/b,/c", or "@file.txt")
  local raw="$1"
  if [[ "$raw" == @* ]]; then
    local f="${raw#@}"
    [[ -f "$f" ]] || { echo "file not found: $f" >&2; exit 1; }
    # Strip comments and blanks; keep one per line
    sed -E 's/[[:space:]]*#.*$//' "$f" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  else
    echo "$raw" | tr ',' '\n' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep -v '^$'
  fi
}

MODELS=$(resolve_list "$MODELS_RAW")
[[ -z "$MODELS" ]] && { echo "no models resolved from: $MODELS_RAW" >&2; exit 1; }

# ------------------------------------------------------------------
# Resolve tasks: --tasks takes priority, else --suite, else default full
# ------------------------------------------------------------------
if [[ -n "$TASKS_RAW" ]]; then
  TASKS=$(resolve_list "$TASKS_RAW")
else
  case "${SUITE:-full}" in
    full)  TASKS=$(echo "$SUITE_FULL"  | tr ',' '\n') ;;
    smoke) TASKS=$(echo "$SUITE_SMOKE" | tr ',' '\n') ;;
    *)     echo "--suite must be full|smoke (got: $SUITE)" >&2; exit 1 ;;
  esac
fi
[[ -z "$TASKS" ]] && { echo "no tasks resolved" >&2; exit 1; }

# ------------------------------------------------------------------
# Container-environment fixes for sbatch from inside Pyxis container.
#
# This script is invoked from inside a container. Two things bite sbatch here:
#   1. The container inherits SLURM_SPANK_* env vars from the parent job which
#      conflict with the new submission's --environment flag. Strip them.
#   2. libjson-c.so.5 (needed by pyxis) is missing from default library path
#      inside the container; we keep a copy in the team wheelhouse.
# ------------------------------------------------------------------
unset $(env | awk -F= '/^SLURM_SPANK/{print $1}') 2>/dev/null || true
export LD_LIBRARY_PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse:${LD_LIBRARY_PATH:-}"

# ------------------------------------------------------------------
# Auto-load WANDB_API_KEY from ~/.netrc and HF_TOKEN from HF cache.
# /users isn't mounted inside the job container so token files there are
# unreachable from the inner slurm job; we re-export here.
# ------------------------------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
  WANDB_API_KEY=$(awk '/api.wandb.ai/{flag=1;next} flag && /password/{print $2; exit}' "$HOME/.netrc")
  export WANDB_API_KEY
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN=$(<"$HOME/.cache/huggingface/token")
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ------------------------------------------------------------------
# Eval defaults (override via env if needed)
# ------------------------------------------------------------------
TOKENIZER_PATH="${TOKENIZER_PATH:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct}"
# 8192 covers ~85% of reasoning-task generations without truncation. Math/reasoning
# tasks at lower budgets show 30-50% mid-response truncation. MCQ hits EOS well
# before this, so no cost for short-answer tasks.
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=8192,temperature=0}"
# 4 vLLM workers per node = 1 per GH200 GPU (4 GPUs). Per-task SQLite handles
# 4 concurrent writers via WAL with sub-ms lock overhead.
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# WandB config
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-alvor}"
WANDB_PROJECT="${WANDB_PROJECT:-apertus-1p5-eval}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-false}"

if [[ "$ENABLE_WANDB" == "true" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: ENABLE_WANDB=true but WANDB_API_KEY is empty (not in ~/.netrc). Jobs will fail fast."
fi

# ------------------------------------------------------------------
# Logging — Anunay's inner default uses scripts/logs which is group-readable
# but NOT writable for us (signal-53). Use the repo logs dir instead.
# ------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-/capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm/logs}"
mkdir -p "$LOG_DIR"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-${LOG_DIR}/eval_${MODE}_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-${LOG_DIR}/eval_${MODE}_%j.err}"

# ------------------------------------------------------------------
# Mode-specific cache flags
#   fill:     write to per-task SQLite (populate it)
#   readonly: read from per-task SQLite (no writes)
# Both modes use --image-token-cache-local-copy 0 since per-task caches are
# small enough to read directly from shared filesystem (no node-local copy).
# ------------------------------------------------------------------
case "$MODE" in
  fill)
    CACHE_READONLY=0
    CACHE_WRITE_MISSES=1
    CACHE_PRELOAD=0
    ;;
  readonly)
    CACHE_READONLY=1
    CACHE_WRITE_MISSES=0
    CACHE_PRELOAD=1
    ;;
esac

# ------------------------------------------------------------------
# Pretty header
# ------------------------------------------------------------------
echo "========================================"
echo "Apertus eval"
echo "  mode:       $MODE"
echo "  models:     $(echo "$MODELS" | tr '\n' ',' | sed 's/,$//')"
echo "  tasks:      $(echo "$TASKS"  | tr '\n' ',' | sed 's/,$//')"
echo "  cache base: $CACHE_BASE"
echo "  slurm:      $SLURM_TEMPLATE"
echo "  wandb:      $ENABLE_WANDB ($WANDB_ENTITY/$WANDB_PROJECT)"
echo "========================================"

# ------------------------------------------------------------------
# Main loop: one sbatch per (task, model) tuple.
# Per-task cache dir is unique → trivially parallel writes during fill,
# bounded blast radius if any single task's cache is corrupted.
# ------------------------------------------------------------------
while IFS= read -r TASK; do
  [[ -z "$TASK" ]] && continue
  TASK_CACHE_DIR="$CACHE_BASE/$TASK"
  mkdir -p "$TASK_CACHE_DIR"

  while IFS= read -r MODEL_PATH; do
    [[ -z "$MODEL_PATH" ]] && continue
    if [[ ! -d "$MODEL_PATH" ]]; then
      echo "WARNING: model path not found, skipping: $MODEL_PATH" >&2
      continue
    fi

    # Derive a stable model label: parent dir name if path ends in /HF, else basename.
    MODEL_LABEL="$(basename "$MODEL_PATH")"
    [[ "$MODEL_LABEL" == "HF" ]] && MODEL_LABEL="$(basename "$(dirname "$MODEL_PATH")")"
    WANDB_GROUP="${WANDB_GROUP_PREFIX}${MODEL_LABEL}"

    echo "--- submit: task=$TASK  model=$MODEL_LABEL ---"
    echo "    cache: $TASK_CACHE_DIR"

    sbatch \
      --output "$SBATCH_OUTPUT" \
      --error  "$SBATCH_ERROR" \
      "$SLURM_TEMPLATE" \
      --model-path "$MODEL_PATH" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --tasks "$TASK" \
      --num-processes "$NUM_PROCESSES" \
      --batch-size "$BATCH_SIZE" \
      --gen-kwargs "$GEN_KWARGS" \
      --enable-image-token-cache true \
      --image-token-cache-dir "$TASK_CACHE_DIR" \
      --image-token-cache-collision-guard 0 \
      --image-token-cache-local-copy 0 \
      --image-token-cache-preload "$CACHE_PRELOAD" \
      --image-token-cache-readonly "$CACHE_READONLY" \
      --image-token-cache-write-misses "$CACHE_WRITE_MISSES" \
      --enable-wandb "$ENABLE_WANDB" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-group "$WANDB_GROUP" \
      --wandb-log-samples "$WANDB_LOG_SAMPLES" \
      --wandb-api-key "${WANDB_API_KEY:-}"
  done <<< "$MODELS"
done <<< "$TASKS"

echo "========================================"
echo "All submissions complete. Tail logs in: $LOG_DIR"
echo "========================================"

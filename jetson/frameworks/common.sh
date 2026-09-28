# Shared helpers for the scripts in jetson/scripts and jetson/frameworks (sourced, not executed; REPO must be set).
#
# resolve_framework <framework> <size>[-<precision>] sources jetson/models/$MODEL/model.sh and
# jetson/frameworks/<framework>.sh and sets FRAMEWORK, SIZE, PRECISION plus the model's variables
# (MODEL_TAG, HF_REPO, AWQ_REPO, GGUF_REPO, IMAGE_* - see models/qwen2_5_vl/model.sh).
# A framework file defines FW_PRECISIONS (first = default), fw_assets (Hub specs to download),
# fw_setup (sets BACKEND, MODEL_ARGS, appends to DOCKER_ARGS; may override IMAGE) and optionally
# fw_start / fw_stop (for server-based frameworks). It can use hf_snapshot / hf_file below and
# REPO, JETSON, OUT, OUT_REL, HF_CACHE, OFFLINE.

JETSON=$REPO/jetson
MODEL=${MODEL:-qwen2_5_vl}
HF_CACHE=${HF_CACHE:-/opt/hf-cache}
OFFLINE=${OFFLINE:-1}
# Containers run as the calling user; if this group exists it is added so shared caches stay group-writable.
SHARED_GROUP=${SHARED_GROUP:-mlusers}

# Results go to jetson/results/<board> (orin, thor, ...) unless RESULTS_DIR (relative to the repo) is set.
board_name() {
  local model
  model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null)
  case "$model" in
    *Thor*) echo thor ;;
    *Orin*) echo orin ;;
    *) echo "${model:-unknown}" | tr '[:upper:] ' '[:lower:]-' ;;
  esac
}
export RESULTS_DIR=${RESULTS_DIR:-jetson/results/$(board_name)}

shared_group_args() {
  local gid
  gid=$(getent group "$SHARED_GROUP" | cut -d: -f3)
  [ -z "$gid" ] || echo "--group-add $gid"
}

# load_model <size>: source jetson/models/$MODEL/model.sh and resolve the size (sets MODEL_TAG, HF_REPO, ...).
load_model() {
  local model_file=$JETSON/models/$MODEL/model.sh
  [ -f "$model_file" ] || { echo "unknown model '$MODEL' (available: $(ls "$JETSON/models" | tr '\n' ' '))" >&2; return 1; }
  source "$model_file"
  model_resolve "$1"
}

resolve_framework() {
  FRAMEWORK=$1
  local spec=${2,,}
  local fw_file=$JETSON/frameworks/$FRAMEWORK.sh
  if [ ! -f "$fw_file" ]; then
    echo "unknown framework '$FRAMEWORK' (available: $(cd "$JETSON/frameworks" && ls *.sh | grep -v common.sh | sed 's/\.sh$//' | tr '\n' ' '))" >&2
    return 1
  fi
  source "$fw_file"
  SIZE=${spec%%-*}
  PRECISION=${spec#"$SIZE"}
  PRECISION=${PRECISION#-}
  PRECISION=${PRECISION:-${FW_PRECISIONS%% *}}
  load_model "$SIZE" || return 1
  if [[ " $FW_PRECISIONS " != *" $PRECISION "* ]]; then
    echo "precision '$PRECISION' not available for $FRAMEWORK (choose: $FW_PRECISIONS)" >&2
    return 1
  fi
}

# Local snapshot dir for a cached Hub repo when running offline, else the repo id.
# (transformers 4.57.3's tokenizer loader queries the Hub for repo ids even with HF_HUB_OFFLINE=1.)
hf_snapshot() {
  local ref=$HF_CACHE/hub/models--${1//\//--}/refs/main
  if [ "$OFFLINE" = 1 ] && [ -f "$ref" ]; then
    echo "$HF_CACHE/hub/models--${1//\//--}/snapshots/$(cat "$ref")"
  else
    echo "$1"
  fi
}

# Path of one file inside a cached Hub repo snapshot (download it first with download_assets.sh).
hf_file() {
  local ref=$HF_CACHE/hub/models--${1//\//--}/refs/main
  local path=$HF_CACHE/hub/models--${1//\//--}/snapshots/$(cat "$ref" 2>/dev/null)/$2
  if [ ! -f "$path" ]; then
    echo "missing $1/$2 in $HF_CACHE - run jetson/scripts/download_assets.sh $FRAMEWORK $SIZE-$PRECISION" >&2
    return 1
  fi
  echo "$path"
}

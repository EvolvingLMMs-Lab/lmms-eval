#!/bin/bash
# Submit a full XModBench-Lite sweep (6 configs) for one model with a
# resource-aware GPU profile.
#
#   light configs (a2t=0, t2a=2: no video)      -> $LIGHT_GRES
#   heavy configs (a2v=1, t2v=3, v2a=4, v2t=5)  -> $HEAVY_GRES
#
# Defaults suit a ~7B model (qwen2.5-omni): light=1xa5000, heavy=4xa5000
# (total 18 GPU < 24 QOS cap -> all 6 run concurrently). For big models
# (e.g. qwen3-omni 30B) pass larger profiles, e.g.
#   LIGHT_GRES=gpu:a5000:4 HEAVY_GRES=gpu:a5000:4 ./submit_lite.sh ...
#
# Usage:
#   [LIGHT_GRES=..] [HEAVY_GRES=..] ./submit_lite.sh MODEL PRETRAINED ENV [EXTRA]
set -euo pipefail

MODEL=${1:?MODEL}
PRETRAINED=${2:?PRETRAINED}
ENV=${3:?ENV}
EXTRA=${4:-device_map=auto,attn_implementation=flash_attention_2}

LIGHT_GRES=${LIGHT_GRES:-gpu:a5000:1}
HEAVY_GRES=${HEAVY_GRES:-gpu:a5000:4}

COMMON="--export=ALL,MODEL=${MODEL},PRETRAINED=${PRETRAINED},ENV=${ENV},MODEL_ARGS_EXTRA=${EXTRA} \
        --job-name=xl_${MODEL} run_xmod_lite_generic.slurm"

# a2t (0) is the only truly light config: 1 audio condition + text options.
# t2a (2) has 4 audio options (singer_identification = long songs) and OOMs
# on a single 24GB GPU, so it joins the heavy profile.
echo "[$MODEL] light a2t -> $LIGHT_GRES"
sbatch --array=0 --gres="$LIGHT_GRES" $COMMON

echo "[$MODEL] heavy t2a,a2v,t2v,v2a,v2t -> $HEAVY_GRES"
sbatch --array=1,2,3,4,5 --gres="$HEAVY_GRES" $COMMON

echo "Submitted. Watch: squeue -u $USER"

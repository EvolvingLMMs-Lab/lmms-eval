#!/bin/bash
# Generic XModBench-Lite runner. Model-agnostic: pass MODEL / PRETRAINED / ENV
# / MODEL_ARGS_EXTRA via --export. GPU profile (count/type) is set by the
# submitting launcher via `sbatch --gres=...`.
#SBATCH --job-name=xmod_lite
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench_lite/%x_%A_%a.log
#SBATCH --error=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench_lite/%x_%A_%a.log

TASKS=(
    xmod_bench_lite_a2t  # 0  no video  (light)
    xmod_bench_lite_a2v  # 1  has video (heavy)
    xmod_bench_lite_t2a  # 2  no video  (light)
    xmod_bench_lite_t2v  # 3  has video (heavy)
    xmod_bench_lite_v2a  # 4  has video (heavy)
    xmod_bench_lite_v2t  # 5  has video (heavy)
)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

MODEL=${MODEL:?set MODEL}
PRETRAINED=${PRETRAINED:?set PRETRAINED}
ENV=${ENV:?set ENV}
MODEL_ARGS_EXTRA=${MODEL_ARGS_EXTRA:-device_map=auto,attn_implementation=flash_attention_2}

REPO=/home/xwang378/scratch/2025/lmms-eval
export XMODBENCH=/home/xwang378/scratch/2025/AudioBench_data
export HF_HOME=/scratch/xwang378/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$NUMBA_CACHE_DIR" "$HF_HOME/hub" "$HF_HOME/datasets"

echo "=== Job $SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID model=$MODEL task=$TASK ==="
echo "=== Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES ==="
date

OUT="$REPO/logs/xmod_bench_lite/results_${MODEL}"
mkdir -p "$OUT"
cd "$REPO"
module load conda
conda activate "$ENV"

python -m lmms_eval \
    --model "$MODEL" \
    --model_args pretrained=${PRETRAINED},${MODEL_ARGS_EXTRA} \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$OUT" \
    --log_samples \
    --log_samples_suffix "$TASK"

echo "=== Done: $TASK ==="
date

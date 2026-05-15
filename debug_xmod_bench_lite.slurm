#!/bin/bash
#SBATCH --job-name=xmod_debug
#SBATCH --gres=gpu:6000_ada:4
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench_lite/debug_%x_%j.log
#SBATCH --error=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench_lite/debug_%x_%j.log

set -e
MODEL=${MODEL:-qwen2_5_omni}
PRETRAINED=${PRETRAINED:-Qwen/Qwen2.5-Omni-7B}
ENV=${ENV:-qwenomni3}
TASK=${TASK:-xmod_bench_lite_a2t}

REPO=/home/xwang378/scratch/2025/lmms-eval
export XMODBENCH=/home/xwang378/scratch/2025/AudioBench_data

export HF_HOME=/scratch/xwang378/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$NUMBA_CACHE_DIR" "$HF_HOME/hub" "$HF_HOME/datasets"

echo "=== Debug job $SLURM_JOB_ID  model=$MODEL  env=$ENV  task=$TASK ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
date

mkdir -p "$REPO/logs/xmod_bench_lite/debug_results"
cd "$REPO"

module load conda
conda activate "$ENV"

python -m lmms_eval \
    --model "$MODEL" \
    --model_args pretrained=$PRETRAINED,device_map=auto,attn_implementation=flash_attention_2 \
    --tasks "$TASK" \
    --limit 8 \
    --batch_size 1 \
    --output_path "$REPO/logs/xmod_bench_lite/debug_results" \
    --log_samples \
    --log_samples_suffix "debug_${MODEL}_${TASK}"

echo "=== Debug done ==="
date

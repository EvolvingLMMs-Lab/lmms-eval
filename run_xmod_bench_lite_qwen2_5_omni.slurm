#!/bin/bash
#SBATCH --job-name=xmod_bench_lite_qwen
#SBATCH --array=0-5
#SBATCH --gres=gpu:6000_ada:4
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench_lite/%x_%a.log
#SBATCH --error=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench_lite/%x_%a.log

TASKS=(
    xmod_bench_lite_a2t  # 1000
    xmod_bench_lite_a2v  # 1000
    xmod_bench_lite_t2a  # 1000
    xmod_bench_lite_t2v  # 1000
    xmod_bench_lite_v2a  # 1000
    xmod_bench_lite_v2t  # 1000
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/xwang378/scratch/2025/lmms-eval
export XMODBENCH=/home/xwang378/scratch/2025/AudioBench_data

# Keep all HF and JIT caches on /scratch — $HOME quota is full.
export HF_HOME=/scratch/xwang378/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$NUMBA_CACHE_DIR" "$HF_HOME/hub" "$HF_HOME/datasets"

echo "=== Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  task=$TASK ==="
echo "=== Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES ==="
date

mkdir -p "$REPO/logs/xmod_bench_lite" "$REPO/logs/xmod_bench_lite/results"

cd "$REPO"
module load conda
conda activate qwenomni3

python -m lmms_eval \
    --model qwen2_5_omni_interleave \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B,device_map=auto,attn_implementation=flash_attention_2 \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$REPO/logs/xmod_bench_lite/results" \
    --log_samples \
    --log_samples_suffix "$TASK"

echo "=== Done: $TASK ==="
date

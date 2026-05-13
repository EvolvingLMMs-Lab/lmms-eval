#!/bin/bash
#SBATCH --job-name=xmod_bench_lite_qwen
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
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

echo "=== Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  task=$TASK ==="
echo "=== Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES ==="
date

mkdir -p "$REPO/logs/xmod_bench_lite" "$REPO/logs/xmod_bench_lite/results"

cd "$REPO"
module load conda
conda activate qwenomni3

python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$REPO/logs/xmod_bench_lite/results" \
    --log_samples \
    --log_samples_suffix "$TASK"

echo "=== Done: $TASK ==="
date

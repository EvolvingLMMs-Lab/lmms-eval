#!/bin/bash
#SBATCH --job-name=xmod_bench_qwen2_5_omni
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench/%x_%a.log
#SBATCH --error=/home/xwang378/scratch/2025/lmms-eval/logs/xmod_bench/%x_%a.log
# Optional: uncomment and fill in if your cluster requires these
# #SBATCH --partition=gpu
# #SBATCH --account=mylab

# Full XModBench (61,320 samples). For the 6k Lite version use:
#   --array=0-5   and   TASKS=( xmod_bench_lite_{a2t,a2v,t2a,t2v,v2a,v2t} )
TASKS=(
    xmod_bench_audio_text    # 10,220
    xmod_bench_text_audio    # 10,220
    xmod_bench_audio_image   #  7,189
    xmod_bench_image_audio   #  7,189
    xmod_bench_image_text    #  7,189
    xmod_bench_text_image    #  7,189
    xmod_bench_audio_video   #  3,031
    xmod_bench_text_video    #  3,031
    xmod_bench_video_audio   #  3,031
    xmod_bench_video_text    #  3,031
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/xwang378/scratch/2025/lmms-eval
# Point XMODBENCH at the staging directory whose layout matches the HF repo
# (Data/, tasks/ at root). Unset to fall back to snapshot_download of
# RyanWW/XModBench under ~/.cache/huggingface/.
export XMODBENCH=/home/xwang378/scratch/2025/AudioBench_data

echo "=== Job $SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  task=$TASK ==="
echo "=== Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES ==="
date

mkdir -p "$REPO/logs/xmod_bench" "$REPO/logs/xmod_bench/results"

cd "$REPO"
module load conda
conda activate qwenomni3

python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$REPO/logs/xmod_bench/results" \
    --log_samples \
    --log_samples_suffix "$TASK"

echo "=== Done: $TASK ==="
date

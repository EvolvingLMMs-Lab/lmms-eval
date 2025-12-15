#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=eval-emu3
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_emu3_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_emu3_%j.err

TASKS="ai2d,vqav2,mme,mmmu"
BATCH_SIZE=1
MODEL="emu3"
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/emu3_results"
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval
OFFLINE_DATASETS="true"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Automatically get number of available GPUS to configure accelerate properöy
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Number of GPUs available: $NUM_GPUS"
echo "GPUS: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "========================================"
echo "MLLM Evaluation Configuration"
echo "========================================"
echo "Model:                ${MODEL}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Num GPU:              ${NUM_GPUS}"
echo "Output-Path:          ${RES_PATH}"
echo "Eval-Repo-Location:   ${EVAL_DIR}"
echo "========================================"
echo ""

echo "###Setup Packages & environment#####"
echo ""
echo "<<Current Packages>>"
pip list
echo ""

echo "📦 Setting up lmms-eval package..."
cd "$EVAL_DIR" || exit
pip uninstall jupyterlab -y  # Uninstall conflicting package
pip install -e .

#echo "📦 Install Emu3 specific versions"
#pip install -e .[emu3]

echo "<<New Packages>>"
pip list
echo ""


accelerate launch -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}"

echo "End Reached"
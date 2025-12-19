#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=eval-llama-emu3
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_llama_emu3_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_llama_emu3_%j.err

TASKS="mmmu_val_group_img,vqav2_val" #,pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
BATCH_SIZE=1
MODEL="llama_emu3"
DEBUG_ARGS="debug_samples=true,num_debug_samples=10" # ex for debug outputs

PRETRAINED="/users/rkreft/MLLM-infra01-folder/rkreft/Meg-Runs/image-extension/llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED-eqsample-loss-SEED28/HF"
TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"
EMU3_MIN_PIXELS="${EMU3_MIN_PIXELS:-262144}"     # 512*512
EMU3_MAX_PIXELS="${EMU3_MAX_PIXELS:-1048576}"    # 1024*1024

RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/llama_emu3_results"
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

# ========== Build model_args ==========
MODEL_ARGS="model_descriptor=$PRETRAINED"
MODEL_ARGS="$MODEL_ARGS,tokenizer_path=$TOKENIZER_PATH"
MODEL_ARGS="$MODEL_ARGS,emu3_min_pixels=$EMU3_MIN_PIXELS"
MODEL_ARGS="$MODEL_ARGS,emu3_max_pixels=$EMU3_MAX_PIXELS"
MODEL_ARGS="$MODEL_ARGS,$DEBUG_ARGS"

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
echo "Model-Args            ${MODEL_ARGS}"
echo "========================================"
echo ""

# Run evaluation command
echo "Running LM evaluation..."
# Conditionally set/unset offline mode
if [ "$OFFLINE_DATASETS" = "true" ]; then
    export HF_DATASETS_OFFLINE=1
else
    unset HF_DATASETS_OFFLINE
fi

echo "###Setup Packages & environment#####"
echo ""
echo "<<Current Packages>>"
pip list
echo ""

echo "📦 Setting up lmms-eval package..."
cd "$EVAL_DIR" || exit
pip uninstall jupyterlab -y  # Uninstall conflicting package
pip install -e .

echo "📦 Installing task-specific dependencies..."
source "${EVAL_DIR}/examples/install_task_deps.sh" "${TASKS}" "${EVAL_DIR}"

echo "📦 Install Emu3 specific versions"
pip install -e .[emu3]

echo "<<New Packages>>"
pip list
echo ""

# Construct model_args parameter if MODEL_ARGS is non-empty
MODEL_ARGS_PARAM=""
if [ -n "${MODEL_ARGS}" ]; then
    MODEL_ARGS_PARAM="--model_args ${MODEL_ARGS}"
fi

accelerate launch -m lmms_eval \
    --model "${MODEL}" \
    ${MODEL_ARGS_PARAM} \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}"

echo "End Reached"
#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --job-name=eval-apertus-emu3-simple
#SBATCH --environment=emu3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=285
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_apertus_emu3_simple_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/lmms-eval/logs/eval_apertus_emu3_simple_%j.err

# Unset SSL_CERT_FILE to avoid FileNotFoundError inside NGC container
# (host path /etc/ssl/ca-bundle.pem doesn't exist in container)
unset SSL_CERT_FILE

TASKS="mmmu_val_group_img,vqav2_val" #,pope,gqa,vqav2_val,mmmu_test_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa
BATCH_SIZE=1
MODEL="apertus_emu3_simple"
MODEL_PATH=""        # Required: path to HF model or HF hub identifier (e.g. "org/model-name")
TOKENIZER_PATH=""    # Required: path to tokenizer or HF hub identifier (defaults to MODEL_PATH)
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/apertus_emu3_simple_results"
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval
OFFLINE_DATASETS="true"

# Emu3 default pixel ranges (use arithmetic expressions like 700*700)
EMU_MIN_PIXELS=$((128*128))
EMU_MAX_PIXELS=$((700*700))

# Debug mode (disabled by default)
DEBUG_SAMPLES="false"
NUM_DEBUG_SAMPLES=5

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
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
        --emu-min-pixels)
            EMU_MIN_PIXELS=$(($2))
            shift 2
            ;;
        --emu-max-pixels)
            EMU_MAX_PIXELS=$(($2))
            shift 2
            ;;
        --debug)
            DEBUG_SAMPLES="true"
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                NUM_DEBUG_SAMPLES="$2"
                shift
            fi
            shift
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

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model-path is required"
    echo "Usage: sbatch $0 --model-path /path/to/model [OPTIONS]"
    exit 1
fi

# Default tokenizer to model path if not set
if [ -z "$TOKENIZER_PATH" ]; then
    TOKENIZER_PATH="$MODEL_PATH"
fi

# Automatically get number of available GPUS to configure accelerate properly
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Number of GPUs available: $NUM_GPUS"
echo "GPUS: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "========================================"
echo "MLLM Evaluation Configuration"
echo "========================================"
echo "Model:                ${MODEL}"
echo "Model path:           ${MODEL_PATH}"
echo "Tokenizer path:       ${TOKENIZER_PATH}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Emu min pixels:       ${EMU_MIN_PIXELS}"
echo "Emu max pixels:       ${EMU_MAX_PIXELS}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Debug samples:        ${DEBUG_SAMPLES} (${NUM_DEBUG_SAMPLES})"
echo "Num GPU:              ${NUM_GPUS}"
echo "Output-Path:          ${RES_PATH}"
echo "Eval-Repo-Location:   ${EVAL_DIR}"
echo "========================================"
echo ""

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

echo "Setting up lmms-eval package..."
cd "$EVAL_DIR" || exit
pip uninstall jupyterlab -y  # Uninstall conflicting package
pip install -e .
pip install -e ".[metrics]"

echo "Installing task-specific dependencies..."
source "${EVAL_DIR}/examples/install_task_deps.sh" "${TASKS}" "${EVAL_DIR}"

echo "<<New Packages>>"
pip list
echo ""

# Build model_args with required parameters
MODEL_ARGS="model_descriptor=${MODEL_PATH},tokenizer_path=${TOKENIZER_PATH}"
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LENGTH}"
fi
if [ -n "$EMU_MIN_PIXELS" ]; then
    MODEL_ARGS="${MODEL_ARGS},emu_min_pixels=${EMU_MIN_PIXELS}"
fi
if [ -n "$EMU_MAX_PIXELS" ]; then
    MODEL_ARGS="${MODEL_ARGS},emu_max_pixels=${EMU_MAX_PIXELS}"
fi
if [ "$DEBUG_SAMPLES" = "true" ]; then
    MODEL_ARGS="${MODEL_ARGS},debug_samples=true,num_debug_samples=${NUM_DEBUG_SAMPLES}"
fi

echo "Running LM evaluation..."
accelerate launch -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}"

echo "End Reached"

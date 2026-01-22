#!/bin/bash
# MIO Model - General Evaluation Script
#
# This script evaluates MIO-7B on any task
# Supports both understanding and generation tasks
#
# Usage:
#   bash mio.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH] [MASTER_PORT] [HF_REPO]
#
# Examples:
#   # ChartQA (10 samples)
#   bash mio.sh "0" "chartqa100" "./logs/chartqa"
#   /home/aiscuser/lmms-eval/g2u/mio.sh
#   # VQAv2
#   bash mio.sh "0" "vqav2_val_lite" "./logs/vqav2"
#
#   # MMBench
#   bash mio.sh "0" "mmbench_en_dev" "./logs/mmbench"
#   bash uniworld_general.sh "2" "chartqa100" "./logs/chartqa" "./models/UniWorld-V1" --verbosity DEBUG
#   # COCO Captioning
#   bash mio.sh "0" "coco_cap_val" "./logs/coco"


#   bash /home/aiscuser/lmms-eval/g2u/mio.sh "4" "chartqa100" "./logs/test"
# illusionbench_arshia_test
# bash /home/aiscuser/lmms-eval/g2u/mio.sh "4" "illusionbench_arshia_test" "./logs/test"
#   # Multiple GPUs
#   bash mio.sh "0,1" "chartqa_test_human" "./logs/chartqa"
#
#   # With custom port
#   bash mio.sh "0" "chartqa100" "./logs/chartqa" "m-a-p/MIO-7B-Instruct" "29604"
#
#   # With HuggingFace upload
#   bash mio.sh "0" "chartqa100" "./logs/chartqa" "m-a-p/MIO-7B-Instruct" "29603" "username/repo-name"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"chartqa100"}
OUTPUT_PATH=${3:-"./logs/mio_${TASK}"}
MODEL_PATH=${4:-"m-a-p/MIO-7B-Instruct"}
MASTER_PORT_ARG=${5:-"29603"}
HF_REPO=${6:-""}  # Optional: HuggingFace repo to upload logs (e.g., "username/repo-name")
BATCH_SIZE=1

# ============ Check MIO Repository ============
if [ ! -d "../MIO" ] && [ ! -d "MIO" ]; then
    echo "❌ Error: MIO repository not found!"
    echo ""
    echo "Please clone MIO repository:"
    echo "  cd .."
    echo "  git clone https://github.com/MIO-Team/MIO.git"
    echo "  cd MIO"
    echo "  pip install -r requirements.txt"
    echo "  cd ../lmms-eval"
    exit 1
fi

echo "✅ MIO repository found"

# ============ Model Arguments ============
# MIO generation config (optimized for image understanding tasks)
MODEL_ARGS="pretrained=${MODEL_PATH}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=512"
MODEL_ARGS="${MODEL_ARGS},num_beams=5"
MODEL_ARGS="${MODEL_ARGS},do_sample=False"
MODEL_ARGS="${MODEL_ARGS},temperature=1.0"
MODEL_ARGS="${MODEL_ARGS},top_p=0.7"
MODEL_ARGS="${MODEL_ARGS},repetition_penalty=1.0"

# ============ Environment Setup ============
export LD_LIBRARY_PATH=/home/aiscuser/cuda_compat:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=${MASTER_PORT_ARG}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1

# ============ Print Configuration ============
echo "======================================"
echo "MIO - General Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Model Args:    ${MODEL_ARGS}"
echo "Master Port:   ${MASTER_PORT}"
if [ -n "${HF_REPO}" ]; then
    echo "HF Upload:     ${HF_REPO}"
fi
echo "======================================"
echo ""

# ============ Run Evaluation ============
python -m lmms_eval \
  --model mio \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  --log_samples_suffix mio_${TASK} \
  --verbosity INFO

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "======================================"

# ============ Upload to HuggingFace (Optional) ============
if [ -n "${HF_REPO}" ]; then
    echo ""
    echo "======================================"
    echo "Uploading logs to HuggingFace..."
    echo "Repository: ${HF_REPO}"
    echo "======================================"

    # Check if huggingface_hub is installed
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "⚠️  Warning: huggingface_hub not installed. Skipping upload."
        echo "Install with: pip install huggingface_hub"
    else
        # Upload the entire output directory to HuggingFace
        python -c "
from huggingface_hub import HfApi
import os

api = HfApi()
output_path = '${OUTPUT_PATH}'
repo_id = '${HF_REPO}'
task_name = '${TASK}'

# Upload all files in the output directory
try:
    api.upload_folder(
        folder_path=output_path,
        repo_id=repo_id,
        path_in_repo=f'logs/{task_name}',
        repo_type='dataset',
        commit_message=f'Upload evaluation logs for {task_name}'
    )
    print(f'✅ Successfully uploaded logs to {repo_id}')
except Exception as e:
    print(f'❌ Failed to upload: {e}')
"
    fi
fi

#!/bin/bash
# MIO Model - General Evaluation Script
#
# This script evaluates MIO-7B on any task
# Supports both understanding and generation tasks
#
# Usage:
#   bash mio.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH]
#
# Examples:
#   # ChartQA (10 samples)
#   bash mio.sh "0" "chartqa100" "./logs/chartqa"
#   
#   # VQAv2
#   bash mio.sh "0" "vqav2_val_lite" "./logs/vqav2"
#   
#   # MMBench
#   bash mio.sh "0" "mmbench_en_dev" "./logs/mmbench"
#   
#   # COCO Captioning
#   bash mio.sh "0" "coco_cap_val" "./logs/coco"
#   
#   # Multiple GPUs
#   bash mio.sh "0,1" "chartqa_test_human" "./logs/chartqa"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"chartqa100"}
OUTPUT_PATH=${3:-"./logs/mio_${TASK}"}
MODEL_PATH=${4:-"m-a-p/MIO-7B-Instruct"}
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
MODEL_ARGS="${MODEL_ARGS},num_beams=5"
MODEL_ARGS="${MODEL_ARGS},do_sample=False"
MODEL_ARGS="${MODEL_ARGS},temperature=1.0"
MODEL_ARGS="${MODEL_ARGS},top_p=0.7"
MODEL_ARGS="${MODEL_ARGS},repetition_penalty=1.0"

# Detect task type and adjust parameters
if [[ "$TASK" == *"caption"* ]] || [[ "$TASK" == *"coco"* ]]; then
    # Image captioning: longer outputs
    MODEL_ARGS="${MODEL_ARGS},max_new_tokens=128"
    echo "📝 Captioning task detected - using max_new_tokens=128"
elif [[ "$TASK" == *"vqa"* ]] || [[ "$TASK" == *"chart"* ]]; then
    # VQA/Chart: shorter answers
    MODEL_ARGS="${MODEL_ARGS},max_new_tokens=64"
    echo "❓ VQA task detected - using max_new_tokens=64"
else
    # Default for other tasks
    MODEL_ARGS="${MODEL_ARGS},max_new_tokens=512"
    echo "📊 General task - using max_new_tokens=512"
fi

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=29603
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

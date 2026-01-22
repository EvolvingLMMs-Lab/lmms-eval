#!/bin/bash
# Test Bagel on Uni-MMMU Maze Visual CoT with Quantization (Memory-Efficient)

# GPU ID (default: 0)
GPU_ID=${1:-0}

# Quantization mode: 4bit or 8bit
QUANT_MODE=${2:-"4bit"}

# Set environment variables
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

echo "Running on GPU: ${GPU_ID}"
echo "Quantization: ${QUANT_MODE}"

if [ "$QUANT_MODE" = "4bit" ]; then
    MODEL_ARGS="pretrained=./BAGEL-7B-MoT,mode=generation,load_in_4bit=True,output_image_dir=/mnt/data/bagel_maze/images"
elif [ "$QUANT_MODE" = "8bit" ]; then
    MODEL_ARGS="pretrained=./BAGEL-7B-MoT,mode=generation,load_in_8bit=True,output_image_dir=/mnt/data/bagel_maze/images"
else
    echo "Invalid quantization mode: $QUANT_MODE (use 4bit or 8bit)"
    exit 1
fi

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model bagel \
  --model_args ${MODEL_ARGS} \
  --tasks uni_mmmu_maze100_visual_cot \
  --batch_size 1 \
  --output_path /mnt/data/bagel_maze \
  --log_samples \
  --verbosity INFO

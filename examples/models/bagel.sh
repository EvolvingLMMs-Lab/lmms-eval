#!/bin/bash

# Bagel Model Evaluation Script
#
# This script demonstrates how to run lmms-eval with the Bagel multimodal model
# for text-to-image generation tasks.
#
# Prerequisites:
#   1. Clone Bagel repository at lmms-eval root:
#      cd /path/to/lmms-eval
#      git clone https://github.com/ByteDance-Seed/Bagel.git
#
#   2. Model weights can be anywhere (specify via MODEL_PATH below)
#      Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
#
# Usage:
#   bash examples/models/bagel.sh

# Set model path - should point to the model weights directory
# Can be absolute path or relative path
MODEL_PATH=$1
export GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
TASK=$2

# Run evaluation with BFloat16 (default, full precision)
 accelerate launch -m lmms_eval \
    --model bagel \
    --model_args pretrained=${MODEL_PATH},mode=1 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

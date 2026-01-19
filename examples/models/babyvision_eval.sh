#!/bin/bash

# BabyVision Benchmark Evaluation Script
# Dataset: UnipatAI/BabyVision (388 items)
# Reference: https://github.com/UniPat-AI/BabyVision

export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# LLM Judge configuration (for blank question evaluation)
# Set these environment variables for LLM judge to work
export API_TYPE="${API_TYPE:-openai}"
export MODEL_VERSION="${MODEL_VERSION:-gpt-4o-2024-11-20}"

# Example 1: Evaluate with local model (Qwen2.5-VL)
echo "=== BabyVision Evaluation with Qwen2.5-VL ==="
python3 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056 \
    --tasks babyvision \
    --batch_size 1 \
    --limit 10 \
    --output_path ./logs/babyvision_qwen/ \
    --log_samples \
    --verbosity INFO

# Example 2: Evaluate with OpenAI-compatible API (e.g., OpenRouter)
# Uncomment and set OPENROUTER_API_KEY to use
# echo "=== BabyVision Evaluation with OpenRouter API ==="
# export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set}"
# export OPENAI_API_BASE="https://openrouter.ai/api/v1"
# MODEL_VERSION="google/gemini-2.5-flash"
#
# python3 -m lmms_eval \
#     --model openai_compatible \
#     --model_args model_version=$MODEL_VERSION \
#     --tasks babyvision \
#     --batch_size 1 \
#     --limit 10 \
#     --output_path ./logs/babyvision_openrouter/ \
#     --log_samples \
#     --verbosity INFO

# Notes:
# - BabyVision has 388 items total (use --limit to test subset)
# - Blank questions use LLM judge for semantic matching (requires API_TYPE/MODEL_VERSION)
# - Choice questions use exact letter matching
# - Output includes type-wise and subtype-wise accuracy breakdown

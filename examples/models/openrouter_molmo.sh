#!/bin/bash

# ============================================================================
# OpenRouter API Test Script
# ============================================================================
# This script demonstrates how to use OpenRouter API with lmms-eval
# Model: allenai/molmo-2-8b:free (free tier model)
#
# OpenRouter provides access to various LLM/VLM models through a unified API
# that is compatible with OpenAI's API format.
# ============================================================================

export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# OpenRouter API configuration (reads from ~/.zshrc OPENROUTER_API_KEY)
export OPENAI_API_KEY="${OPENROUTER_API_KEY:?Error: OPENROUTER_API_KEY not set in environment}"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

# Model to use (free tier)
MODEL_VERSION="allenai/molmo-2-8b:free"

# ============================================================================
# Basic evaluation example
# ============================================================================
# Using a small subset (--limit 5) for quick testing
# Tasks: mme (multimodal evaluation)

python3 -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=$MODEL_VERSION \
    --tasks mme \
    --batch_size 1 \
    --limit 5 \
    --output_path ./logs/openrouter_test/ \
    --log_samples \
    --verbosity DEBUG

# ============================================================================
# Notes:
# ============================================================================
# 1. OpenRouter free models have rate limits - use small --limit for testing
# 2. Some models may not support all image formats or features
# 3. Check https://openrouter.ai/models for available models and pricing
# 4. For production use, consider paid models for better rate limits
# ============================================================================

#!/usr/bin/env bash
# Install lmms-eval[audio] and native Transformers (tested with 5.24.0).
# The stock FLEURS task expects a Hugging Face login for dataset access.
python -m lmms_eval \
    --model qwen3_asr \
    --model_args "pretrained=${MODEL:-Qwen/Qwen3-ASR-0.6B-hf},language=en" \
    --tasks fleurs_en \
    --device "${DEVICE:-cuda}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --log_samples --output_path ./logs

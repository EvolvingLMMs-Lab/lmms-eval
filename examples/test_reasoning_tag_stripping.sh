#!/bin/bash

# Example: Testing reasoning tag stripping with Qwen3-VL
# By default, <think>...</think> blocks are stripped from model output before scoring.
# The raw output (with <think> tags) is preserved in log_samples JSONL under "resps".
# The cleaned output (scored) is under "filtered_resps".

python -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,attn_implementation=flash_attention_2 \
    --tasks mme \
    --batch_size 1 \
    --limit 8 \
    --log_samples \
    --output_path ./output_reasoning_test

# To disable reasoning tag stripping:
# python -m lmms_eval \
#     --model qwen3_vl \
#     --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
#     --tasks mme \
#     --batch_size 1 \
#     --limit 8 \
#     --reasoning_tags none \
#     --log_samples \
#     --output_path ./output_no_strip

echo "Check output_reasoning_test/ for JSONL files."
echo "In the JSONL, 'resps' contains raw model output, 'filtered_resps' contains cleaned output."

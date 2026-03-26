#!/bin/bash
# Test script for issue #1260: qwen2_5_vl video metadata fix
# Runs Qwen2.5-VL-7B-Instruct on charades_sta with --limit to verify
# that video metadata (fps, frames_indices) is correctly passed to the
# processor, producing correct second_per_grid_ts values.

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,max_num_frames=64,attn_implementation=flash_attention_2 \
    --tasks temporal_grounding_charades \
    --batch_size 1 \
    --limit 10

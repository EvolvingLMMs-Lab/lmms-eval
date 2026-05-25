export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip install qwen-vl-utils

# ============================================================
# Example 1: frame-sampling video backend (default)
# MLVU-dev with best config (min_pixels = max_pixels = 102400,
# max_num_frames = 384).
# ============================================================
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision2 \
    --model_args=pretrained=lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct,attn_implementation=flash_attention_2,messages_format=timestamp,max_new_tokens=16,fps=1,max_num_frames=384,min_pixels=102400,max_pixels=102400 \
    --tasks=mlvu_dev \
    --batch_size=1

# ============================================================
# Example 2: codec video backend (recommended for long videos)
# ============================================================
# The codec backend replaces uniform frame sampling with codec-aware
# canvas packing driven by motion vectors / bit-cost. It typically
# yields stronger long-video accuracy at the same token budget.
#
# Extra requirements (only needed for `use_codec=true`):
#   pip install codec-video-prep opencv-python
#   ffmpeg 4.4.x - 7.x on PATH  (verify: `ffmpeg -version`)
#
# All canvas extraction / caching / chat-template rewriting happens
# inside the model's processor via trust_remote_code; the adapter
# only forwards the video URL.
#
# Optional cache location (defaults to $HF_HOME/online_codec):
# export ONLINE_CODEC_CACHE_DIR=/path/to/codec_cache

accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision2 \
    --model_args=pretrained=lmms-lab-encoder/LLaVA-OneVision2-8B-Instruct,attn_implementation=flash_attention_2,messages_format=timestamp,max_new_tokens=16,use_codec=true,codec_target_canvas=32,codec_group_size=32,codec_images_per_group=4,codec_patch=14,max_pixels=150000 \
    --tasks=videomme_long \
    --batch_size=1

# codec model_args reference:
#   use_codec=true               : enable codec backend
#   codec_target_canvas=32       : number of canvases per video (more -> more tokens)
#   codec_group_size=32          : frames per readiness group (cv-preinfer knob)
#   codec_images_per_group=4     : canvases produced per group
#   codec_patch=14               : ViT patch size; must match the model
#   max_pixels=150000            : per-canvas pixel budget (also fed to image_processor)
#   codec_cache_root=/some/dir   : override the on-disk canvas cache root
#                                  (otherwise uses $ONLINE_CODEC_CACHE_DIR or
#                                   $HF_HOME/online_codec)
#
# Any codec_* kwarg you omit falls back to the default in the model
# repo's preprocessor_config.json ("codec": {...} block).

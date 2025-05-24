export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_32B \
    --output_path ./logs/
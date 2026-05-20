export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip install qwen-vl-utils

# Example: MLVU-dev with best config (min_pixels = max_pixels = 102400, max_num_frames = 384)
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision2 \
    --model_args=pretrained=lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct,attn_implementation=flash_attention_2,messages_format=timestamp,max_new_tokens=16,fps=1,max_num_frames=384,min_pixels=102400,max_pixels=102400 \
    --tasks=mlvu_dev \
    --batch_size=1

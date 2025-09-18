export HF_HOME="~/.cache/huggingface"
export HF_HOME=/vlm/huggingface

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,attn_implementation=flash_attention_2,max_pixels=3240000 \
    --tasks=mmerealworld,mmerealworld_cn,chartqa,docvqa_val,infovqa_val,mmstar,ocrbench \
    --batch_size=1

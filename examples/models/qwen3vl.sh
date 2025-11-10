export HF_HOME="~/.cache/huggingface"
# pip3 install transformers==4.57.1 (Qwen3VL models)
# pip3 install ".[qwen]" (for Qwen's dependencies)

# Exmaple with Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct 

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_vl \
    --model_args=pretrained=Qwen/Qwen3-VL-4B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks "mmmu_val,mmbench_en_dev,ocrbench,realworldqa,mmstar" \
    --batch_size 1
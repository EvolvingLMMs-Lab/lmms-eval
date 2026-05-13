# export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# Penguin-VL inference recommends transformers==4.51.3

# Example with Penguin-VL-8B: https://huggingface.co/tencent/Penguin-VL-8B

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model penguinvl \
    --model_args=pretrained=tencent/Penguin-VL-8B,attn_implementation=flash_attention_2,dtype=bfloat16 \
    --tasks "ai2d,mmmu_pro_standard,ocrbench,videomme,longvideobench_val_v" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix penguinvl \
    --verbosity DEBUG \
    --output_path ./logs/
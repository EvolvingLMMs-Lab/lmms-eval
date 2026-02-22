export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

python3 -m lmms_eval \
    --model=llava \
    --model_args=pretrained=lmms-lab/llava-next-72b,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
    --tasks=pope,vizwiz_vqa_val,scienceqa_img \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix=llava_qwen \
    --output_path="./logs/" \
    --wandb_args=project=lmms-eval,job_type=eval,entity=llava-vl
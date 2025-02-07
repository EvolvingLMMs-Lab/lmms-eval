#!/bin/bash
export OPENAI_API_KEY=sk-testmyllm
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_URL=http://0.0.0.0:23335/v1/chat/completions
export GPU=$(nvidia-smi --list-gpus | wc -l)
source /mnt/code/users/liamding/tools/conda_install/anaconda3/bin/activate vlm_llava_lmdeploy
nohup lmdeploy serve api_server /mnt/data/huggingface/models/Qwen2/Qwen2-7B-Instruct --server-port 23335 --tp 4 --log-level INFO --model-name gpt-3.5-turbo --cache-max-entry-count 0.1 --api-keys sk-testmyllm >lmdeploy_deploy.log &
sleep 10
source /mnt/code/users/liamding/tools/conda_install/anaconda3/bin/activate lmms_eval
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    --main_process_port=12555 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/mnt/data/users/zhangyuqi/huggingface/vlm_models/llava-v1.5-7b" \
    --tasks hrbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_7b_hrbench_debug \
    --output_path ./logs/
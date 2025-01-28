
accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model gpt4v \
    --model_args model_version=gpt-4o-mini,modality=image \
    --tasks mmbench_cn_dev \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmbench_cn_dev \
    --output_path ./logs/ \
    --limit 10


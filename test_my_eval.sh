
accelerate launch --num_processes 2 --main_process_port 30000 -m lmms_eval \
    --model gpt \
    --model_args model_version=gpt-4o-mini \
    --tasks genai_rqa \
    --batch_size auto \
    --log_samples \
    --output_path ./logs/ \
    --limit 10

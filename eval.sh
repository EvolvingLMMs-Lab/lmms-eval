
accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model gpt \
    --model_args model_version=gpt-4o-mini,api_url=http://localhost:9621/v1/chat/completions\
    --tasks paper_rqa_o1_60000 \
    --batch_size auto \
    --log_samples \
    --output_path ./logs/ \
    --wandb_args project=test_log,name=lightrag_mix \
    --limit 5


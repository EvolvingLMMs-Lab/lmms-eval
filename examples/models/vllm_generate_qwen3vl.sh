

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model vllm_generate \
    --model_args model=Qwen/Qwen3-VL-4B-Instruct,tensor_parallel_size=1,disable_log_stats=True \
    --tasks mmmu_val \
    --batch_size 32 \
    --log_samples \
    --output_path ./logs --verbosity DEBUG --limit 32
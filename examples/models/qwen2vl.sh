# Run and exactly reproduce qwen2vl results!
# mme as an example
pip3 install qwen_vl_utils
accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=2359296 \
    --tasks mme,gsm8k_cot_self_consistency,mmmu_val_reasoning,mmmu_val_pass64 \
    --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/
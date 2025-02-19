
# pip3 install vllm
# pip3 install qwen_vl_utils

# cd ~/prod/lmms-eval-public
# pip3 install -e .

python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=Qwen/Qwen2-VL-7B-Instruct,tensor_parallel_size=4 \
    --tasks mme,gsm8k_cot_self_consistency,mmmu_val \
    --batch_size 64 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs
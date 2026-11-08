# uv pip install -e ".[qwen]" (for Qwen's dependencies)
# uv pip install --upgrade transformers  # qwen4_exp needs transformers>=5.22
# Qwen3.8-Flash-Next: 180B parameters in total (6B active, plus a 51B n-gram embedding).

MODEL='Qwen/Qwen3.8-Flash-Next-FP8'
TASKS='mathvision_testmini'
MAX_PIXELS=$((2048 * 32 * 32))
MIN_PIXELS=$((256 * 32 * 32))
OUTPUT_PATH='./logs/qwen3_8_flash_next_mathvision/'

# Thinking mode is on by default; the recommended sampling is temperature=1.0, top_p=0.95, top_k=20.
# HF backend: needs GPUs that can hold the n-gram embedding (48 GiB in FP8) twice during loading,
# so it does not load on 80 GB GPUs.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_8_flash_next \
    --model_args pretrained=$MODEL,device_map=auto,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS \
    --gen_kwargs max_new_tokens=32768,temperature=1.0,top_p=0.95,top_k=20 \
    --tasks $TASKS \
    --log_samples \
    --output_path $OUTPUT_PATH \
    --batch_size 1

# vLLM backend on 2x 80 GB GPUs: vLLM keeps the n-gram embedding in CPU memory (engram cpu_offload, on by default).
# The n-gram embedding needs pipeline_parallel_size=1. Use expert parallelism: plain tensor parallelism
# splits the MoE layers below the 128-wide FP8 quantization blocks. On 4 or 8 GPUs, raise tensor_parallel_size.
# FlashInfer compiles kernels at startup, so CUDA_HOME must point to a CUDA toolkit that matches PyTorch.
# python -m lmms_eval \
#     --model vllm \
#     --model_args model=$MODEL,tensor_parallel_size=2,enable_expert_parallel=True,moe_backend=triton,gpu_memory_utilization=0.95,max_model_len=16384,max_num_seqs=8,disable_custom_all_reduce=True \
#     --gen_kwargs max_new_tokens=12288,temperature=1.0,top_p=0.95,top_k=20 \
#     --tasks $TASKS \
#     --batch_size 8 \
#     --log_samples \
#     --output_path $OUTPUT_PATH

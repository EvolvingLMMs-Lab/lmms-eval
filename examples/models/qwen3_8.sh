# uv pip install -e ".[qwen]" (for Qwen's dependencies)
# uv pip install --upgrade transformers  # qwen4_exp needs a transformers release from Sep 2026 or later
# Qwen3.8-Flash-Next has 180B parameters (6B active). The FP8 checkpoint (186 GB) fits on three 80 GB GPUs.

MODEL='Qwen/Qwen3.8-Flash-Next-FP8'
TASKS='mathvision_testmini'
MAX_PIXELS=$((2048 * 32 * 32))
MIN_PIXELS=$((256 * 32 * 32))
BATCH_SIZE=1
OUTPUT_PATH='./logs/qwen3_8_mathvision/'

# Thinking mode is on by default; the recommended sampling is temperature=1.0, top_p=0.95, top_k=20.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_8 \
    --model_args pretrained=$MODEL,device_map=auto,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS \
    --gen_kwargs max_new_tokens=32768,temperature=1.0,top_p=0.95,top_k=20 \
    --tasks $TASKS \
    --log_samples \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE

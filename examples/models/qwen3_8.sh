# uv pip install -e ".[qwen]" (for Qwen's dependencies)
# uv pip install --upgrade transformers
# Qwen3.8 dense models use the Qwen3.5 architecture. For Qwen3.8-Flash-Next, see qwen3_8_flash_next.sh.

MODEL='Qwen/Qwen3.8-27B'
TASKS='mathvision_testmini'
MAX_PIXELS=$((2048 * 32 * 32))
MIN_PIXELS=$((256 * 32 * 32))
BATCH_SIZE=1
OUTPUT_PATH='./logs/qwen3_8_mathvision/'

# Thinking mode is on by default; the recommended sampling is temperature=1.0, top_p=0.95, top_k=20.
# Qwen3.8-27B (bf16, 52 GB) fits on one 80 GB GPU.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_8 \
    --model_args pretrained=$MODEL,device_map=auto,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS \
    --gen_kwargs max_new_tokens=32768,temperature=1.0,top_p=0.95,top_k=20 \
    --tasks $TASKS \
    --log_samples \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE

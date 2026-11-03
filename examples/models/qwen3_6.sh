# uv pip install -e ".[qwen]" (for Qwen's dependencies)
# uv pip install --upgrade transformers
# uv pip install flash-linear-attention (optional, makes inference faster)

MODEL='Qwen/Qwen3.6-35B-A3B'
TASKS='ai2d'
MAX_PIXELS=$((2048 * 32 * 32))
MIN_PIXELS=$((256 * 32 * 32))
MAX_NUM_FRAMES=1
BATCH_SIZE=1
OUTPUT_PATH='./logs/qwen3_6_ai2d/'

# AI2D task internally supplies max_new_tokens=16, temperature=0, and do_sample=false.
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model qwen3_6 \
    --model_args pretrained=$MODEL,device_map=auto,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_frames=$MAX_NUM_FRAMES,enable_thinking=false \
    --tasks $TASKS \
    --log_samples \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE

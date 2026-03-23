# pip3 install ".[qwen]" (for Qwen's dependencies)
# pip3 install transformers>=5.3.0 (to get Qwen3_5 models)
# pip3 install flash-linear-attention (optional, make inference faster)

# Smaller Qwen 3.5 models (0.8B and 2B) are non thinking by default (enable_thinking=false)
# Bigger models are thinking by default (enable_thinking=true)

MODEL='Qwen/Qwen3.5-2B'
TASKS='mmstar_oc,ocrbench'
MAX_PIXELS=$((2048 * 32 * 32))
MIN_PIXELS=$((256 * 32 * 32))
MAX_NUM_FRAMES=1
BATCH_SIZE=8
MAX_NEW_TOKENS=2048

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen3_5 \
    --model_args pretrained=$MODEL,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_frames=$MAX_NUM_FRAMES,enable_thinking=false  \
    --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
    --tasks $TASKS \
    --log_samples \
    --output_path ./logs/ \
    --batch_size $BATCH_SIZE

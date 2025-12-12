# EMU3 Chat Model Evaluation - standalone file

# Install dependencies: pip install -e ".[emu3]"
# Or manually: pip install git+https://github.com/baaivision/Emu3.git

export HF_HOME="~/.cache/huggingface"

# Single GPU evaluation
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model emu3 \
    --model_args attn_implementation=flash_attention_2 \
    --tasks mme,ai2d,vqav2,gqa \
    --batch_size 1 \
    --device cuda:0


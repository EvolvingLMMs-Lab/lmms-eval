export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES="1,2,3"
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model emu3 \
    --model_args pretrained="BAAI/Emu3-Chat"\
    --tasks mmbench_en_test \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix emu3_mmb \
    --output_path ./logs/

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model emu3 \
    --model_args pretrained="BAAI/Emu3-Chat"\
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix emu3_pope \
    --output_path ./logs/
TASK=open_asr_tedlium
CKPT_PATH=lmms-lab/Aero-1-Audio
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 30000 -m lmms_eval \
    --model aero \
    --model_args pretrained=$CKPT_PATH,attn_implementation="flash_attention_2" \
    --tasks $TASK \
    --batch_size 32 \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ --verbosity DEBUG

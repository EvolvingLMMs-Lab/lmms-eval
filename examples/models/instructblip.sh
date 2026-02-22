cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;

CKPT_PATH=$1
TASK=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model instructblip \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix instructblip \
    --output_path ./logs/ 
TASK=$1
CKPT_PATH=$2
CONV_TEMPLATE=$3
MODEL_NAME=$4

echo "Task: $TASK"
echo "Checkpoint Path: $CKPT_PATH"
echo "Conversation Template: $CONV_TEMPLATE"
echo "Model Name: $MODEL_NAME"

TASK_SUFFIX="${TASK//,/_}"
echo "Task Suffix: $TASK_SUFFIX"

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs
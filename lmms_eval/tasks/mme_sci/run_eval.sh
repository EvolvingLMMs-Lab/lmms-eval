TASK=$1
CKPT_PATH=$2
CONV_TEMPLATE=$3
MODEL_NAME=$4
HOST="127.0.0.1"
PORT=8001
MODEL_PATH="Qwen2___5-VL-3B-Instruct"
MEM_FRACTION=0.6
TP=1
TIMEOUT=600

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
    --output_path ./logs \
    --launcher_args "name=sglang host=$HOST port=$PORT model=$MODEL_PATH mem_fraction_static=$MEM_FRACTION tp=$TP timeout=$TIMEOUT enable_torch_compile=False enable_cuda_graph=False log_level='warning' log_level_http='warning'"
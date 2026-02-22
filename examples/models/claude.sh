cd /path/to/lmms-eval
python3 -m pip install -e .;

export ANTHROPIC_API_KEY="<YOUR_API_KEY>"

TASK=$1
MODEL_VERSION=$2
MODALITIES=$3
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model claude \
    --model_args model_version=$MODEL_VERSION\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
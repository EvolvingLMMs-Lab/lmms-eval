cd /path/to/lmms-eval
python3 -m pip install -e .;

python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/rese1f/MovieChat.git
mv /path/to/MovieChat/MovieChat /path/to/lmms-eval/lmms_eval/models

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model moviechat \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
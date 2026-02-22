cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/rese1f/aurora.git
mv /path/to/aurora/src/xtuner/xtuner /path/to/lmms-eval/lmms_eval/models/xtuner-aurora

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model auroracap \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
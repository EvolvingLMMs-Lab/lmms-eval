# First you need to fork [`InternVL`](https://github.com/OpenGVLab/InternVL)

cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/InternVL/internvl_chat
python3 -m pip install -e .;

python3 -m pip install flash-attn==2.3.6 --no-build-isolation;


TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model internvl \
    --model_args pretrained="OpenGVLab/InternVL-Chat-V1-5"\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
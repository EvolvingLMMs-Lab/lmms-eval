# You won't need to clone any other repos to run idefics. Making sure your transformers version supports idefics2 would be enough

cd /path/to/lmms-eval
python3 -m pip install -e .

python3 -m pip install transformers --upgrade

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model idefics2 \
    --model_args pretrained=HuggingFaceM4/idefics2-8b \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
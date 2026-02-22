cd /path/to/lmms-eval
python3 -m pip install -e .;

# Notice that you should not leave the folder of LLaMA-VID when calling lmms-eval
# Because they left their processor's config inside the repo
cd /path/to/LLaMA-VID;
python3 -m pip install -e .

python3 -m pip install av sentencepiece;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llama_vid \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
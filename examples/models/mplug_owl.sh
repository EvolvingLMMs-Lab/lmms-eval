cd /path/to/lmms-eval
python3 -m pip install -e .;

# It has to use an old transformers version to run
python3 -m pip install av sentencepiece protobuf==3.20 transformers==4.28.1 einops;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model mplug_owl_video \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

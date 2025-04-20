cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/yfzhang114/SliME.git

cd SliME
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd ..

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model slime \
    --tasks $TASK \
    --model_args pretrained="yifanzhang114/SliME-Llama3-8B,conv_template=llama3,model_name=slime" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
cd /path/to/lmms-eval
python3 -m pip install -e .;


python3 -m pip install flash-attn --no-build-isolation;
python3 -m pip install torchvision einops timm sentencepiece;


TASK=$1
CKPT_PATH=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12380 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
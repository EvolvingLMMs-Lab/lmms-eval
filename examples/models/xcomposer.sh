cd /path/to/lmms-eval
python3 -m pip install -e .;


python3 -m pip install flash-attn --no-build-isolation;
python3 -m pip install torchvision einops timm sentencepiece;

TASK=$1
MODALITY=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

# For Xcomposer2d5
accelerate launch --num_processes 8 --main_process_port 10000 -m lmms_eval \
    --model xcomposer2d5 \
    --model_args pretrained="internlm/internlm-xcomposer2d5-7b",device="cuda",modality=$MODALITY\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

# For Xcomposer-4kHD
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model xcomposer2_4khd \
    --model_args pretrained="internlm/internlm-xcomposer2-4khd-7b" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
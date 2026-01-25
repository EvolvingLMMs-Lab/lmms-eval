export HF_HOME="~/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# bash ./internvl_hf.sh "mme" "/path/to/InternVL3_5-8B-HF"

NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
TASK=$1
CKPT_PATH=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes $NUM_DEVICES --main_process_port 12380 -m lmms_eval \
    --model internvl_hf \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
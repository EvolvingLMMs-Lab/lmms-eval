model_name=$1
task_name=$2
model_args="attn_implementation=\"flash_attention_2\",dtype=torch.bfloat16"

if [[ -z "${OPENAI_API_KEY}" ]]; then 
    echo "Error: OPENAI_API_KEY environment variable is not set." \
    exit 1      
else 
    mkdir logs/${model_name}_${task_name}
    sngpu  --time 105:59:59 --cpu 8 --mem 300000 --gpu 1 \
    --output eval_logs/${model_name}_${task_name}.log -- \
        "source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh &&  \
        conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval && \
        export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH && \
        export OPENAI_API_KEY=$OPENAI_API_KEY && \
        accelerate launch --num_processes 1 \
        --config_file deepspeed_zero3_2workers.yaml -- \
        lmms_eval --model ${model_name} \
            --model_args ${model_args}  \
            --tasks ${task_name} \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "${model_name}_${task_name}" \
            --output_path logs/${model_name}_${task_name}"
fi 

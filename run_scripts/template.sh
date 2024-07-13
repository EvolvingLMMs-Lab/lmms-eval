model_name=$1
task_name=$2
model_args="attn_implementation=flash_attention_2"
logger_name="${model_name}_${task_name}"
if [ -z "$3" ]; then
    echo "No pretrained checkpoint"
else
    echo "Pretrained Checkpoint: $3"
    model_args+=",pretrained=$3"
    logger_name+="_$(basename $3)"
fi

if [[ $model_name == *"llava_hf"* ]]; then
    model_args+=",device_map=auto,dtype=bfloat16,fast_tokenizer=False"
fi

echo $logger_name
echo $model_args

mkdir logs/${logger_name}
# if [[ true ]]; then
if [[ $model_name == *"gpt"* || $model_name == *"claude"* ]]; then
    if [[ -z "${AZURE_API_KEY}" ]]; then 
        echo "Error: AZURE_API_KEY environment variable is not set." \
        exit 1      
    fi

    if [[ -z "${AZURE_ENDPOINT}" ]]; then 
        echo "Error: AZURE_ENDPOINT environment variable is not set." \
        exit 1      
    fi
    export API_TYPE="azure"
    source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh &&  \
        export NCCL_DEBUG=INFO &&  \
        export NCCL_DEBUG_SUBSYS=ALL &&  \
        export NCCL_P2P_LEVEL=NVL &&  \
        export NCCL_SHM_DISABLE=1 &&  \
        export NCCL_IB_DISABLE=1 &&  \
        conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval && \
        export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH && \
        export OPENAI_API_KEY=$OPENAI_API_KEY && \
        lmms_eval --model ${model_name} \
            --model_args ${model_args}  \
            --tasks ${task_name} \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "${logger_name}" \
            --output_path logs/${logger_name}
else
    sngpu  --time 105:59:59 --cpu 8 --mem 300000 --gpu 1 \
    --output eval_logs/${logger_name}.log -- \
        "source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh &&  \
        export NCCL_DEBUG=INFO &&  \
        export NCCL_DEBUG_SUBSYS=ALL &&  \
        export NCCL_P2P_LEVEL=NVL &&  \
        export NCCL_SHM_DISABLE=1 &&  \
        export NCCL_IB_DISABLE=1 &&  \
        conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval && \
        export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH && \
        export OPENAI_API_KEY=$OPENAI_API_KEY && \
        lmms_eval --model ${model_name} \
            --model_args ${model_args}  \
            --tasks ${task_name} \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "${logger_name}" \
            --output_path logs/${logger_name}"
fi


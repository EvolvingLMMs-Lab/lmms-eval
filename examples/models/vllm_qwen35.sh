#!/bin/bash

MODEL="Qwen/Qwen3.5-397B-A17B"
TASKS="mmmu_val,mme"

TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.85
BATCH_SIZE=16
MAX_MODEL_LEN=262144
# Official Qwen3.5 examples use reasoning_parser=qwen3.
# vLLM recipes currently note deepseek_r1 may parse reasoning content more reliably.
# If reasoning blocks are not parsed as expected on your vLLM version, switch to deepseek_r1.
REASONING_PARSER="qwen3"

OUTPUT_PATH="./logs/qwen35_vllm"
LOG_SUFFIX="qwen35_vllm"

CMD="uv run python -m lmms_eval \
    --model vllm \
    --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},reasoning_parser=${REASONING_PARSER} \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --log_samples --log_samples_suffix ${LOG_SUFFIX} \
    --output_path ${OUTPUT_PATH}"

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

#!/bin/bash

MODEL="Qwen/Qwen3.5-397B-A17B"
TASKS="mmmu_val,mme"

TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.85
BATCH_SIZE=16
CONTEXT_LENGTH=262144
# Official Qwen3.5 SGLang launch examples use reasoning_parser=qwen3.
REASONING_PARSER="qwen3"

MAX_PIXELS=1605632
MIN_PIXELS=784
MAX_FRAME_NUM=32
THREADS=16

OUTPUT_PATH="./logs/qwen35_sglang"
LOG_SUFFIX="qwen35_sglang"

CMD="uv run python -m lmms_eval \
    --model sglang \
    --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},context_length=${CONTEXT_LENGTH},reasoning_parser=${REASONING_PARSER},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_frame_num=${MAX_FRAME_NUM},threads=${THREADS} \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --log_samples --log_samples_suffix ${LOG_SUFFIX} \
    --output_path ${OUTPUT_PATH}"

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

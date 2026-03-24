#!/bin/bash
set -euo pipefail

# 1. 自动检测本地资源
LOCAL_GPU_NUM=$(nvidia-smi -L | wc -l)
TP="${TP:-1}"
NUM_BACKENDS=$(( LOCAL_GPU_NUM / TP ))

# --- 日志目录准备 ---
VLLM_LOG_DIR="${VLLM_LOG_DIR:-/mnt/cpfs/yangyicun/vllm_logs/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "${VLLM_LOG_DIR}"

# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# 2. 分布式环境参数适配
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
WORLD_SIZE="${WORLD_SIZE:-8}"  
RANK="${RANK:-0}"              
NPROC_PER_NODE="${NPROC_PER_NODE:-$LOCAL_GPU_NUM}"
NUM_MACHINES=$(( (WORLD_SIZE + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))  
MACHINE_RANK=$(( RANK / NPROC_PER_NODE ))

echo "[INFO][Machine ${MACHINE_RANK}] Global Rank: ${RANK}/${WORLD_SIZE} | Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO][Machine ${MACHINE_RANK}] Logging all outputs to: ${VLLM_LOG_DIR}"

# 3. 激活环境
source /mnt/cpfs/yangyicun/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/cpfs/yangyicun/miniconda3/envs/lmms-eval-vllm

MODEL="${MODEL:-/mnt/cpfs/yangyicun/data/model/Qwen3-VL-8B-Instruct}"
TASKS="${TASKS:-mmmu,scienceqa,vizwiz_vqa_val,chartqa,ai2d,docvqa_val,ocrbench,seedbench,textvqa_val}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/cpfs/yangyicun/eval_result}"
CONCURRENCY="${CONCURRENCY:-1024}" 
VERBOSITY="${VERBOSITY:-DEBUG}" 

# --- 进程管理：确保退出时杀死所有后台 vLLM 实例 ---
PIDS=()
cleanup() {
    echo "[INFO][Machine ${MACHINE_RANK}] Cleaning up vLLM instances (PIDs: ${PIDS[*]})..."
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# 4. 循环启动本地 vLLM 实例
URLS=""
for (( i=0; i<$NUM_BACKENDS; i++ )); do
    PORT=$(( 8001 + i ))
    START_GPU=$(( i * TP ))
    END_GPU=$(( (i + 1) * TP - 1 ))
    
    GPUS=""
    for (( g=$START_GPU; g<=$END_GPU; g++ )); do
        GPUS="${GPUS}${g},"
    done
    GPUS=${GPUS%,}
    
    LOG_FILE="${VLLM_LOG_DIR}/vllm_instance_rank${RANK}_port${PORT}.log"
    
    echo "[INFO][Machine ${MACHINE_RANK}] Starting vLLM on GPUs ${GPUS} at port ${PORT}..."
    
    CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --gpu-memory-utilization 0.8 \
        --port "${PORT}" \
        --mm-encoder-tp-mode data \
        --trust-remote-code \
        --max-num-seqs 512 \
        --enable-prefix-caching \
        > "${LOG_FILE}" 2>&1 &
    
    PIDS+=($!)
    URLS="${URLS}http://localhost:${PORT}/v1;"
done

URLS=${URLS%;}

# 5. 等待本地所有后端就绪
check_vllm() {
    curl -s -o /dev/null -w "%{http_code}" "$1/models" || echo "000"
}

echo "[INFO][Machine ${MACHINE_RANK}] Waiting for backends to be ready..."
IFS=';' read -ra URL_ARRAY <<< "$URLS"
for url in "${URL_ARRAY[@]}"; do
    RETRY_COUNT=0
    while [ "$(check_vllm $url)" != "200" ]; do
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ ${RETRY_COUNT} -ge 360 ]; then
            echo "[ERROR] vLLM at $url failed to start."
            exit 1
        fi
    done
done

echo "[INFO][Machine ${MACHINE_RANK}] All local vLLM instances are UP."

# 6. 使用 accelerate launch 提交评测
LMMS_LOG_FILE="${VLLM_LOG_DIR}/lmms_eval_rank${RANK}.log"
echo "[INFO][Machine ${MACHINE_RANK}] Launching LMMS-Eval. Log: ${LMMS_LOG_FILE}"

accelerate launch \
    --num_processes "${WORLD_SIZE}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision "no" \
    --dynamo_backend "no" \
    -m lmms_eval \
    --model openai \
    --model_args "model_version=${MODEL},base_url=${URLS},api_key=EMPTY,num_concurrent=${CONCURRENCY},adaptive_concurrency=False,adaptive_max_concurrency=128" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --output_path "${OUTPUT_PATH}" \
    --log_samples \
    --verbosity "${VERBOSITY}" \
    > "${LMMS_LOG_FILE}" 2>&1

echo "[INFO][Machine ${MACHINE_RANK}] Evaluation Task Completed Successfully."

# --model_args "model_version=${MODEL},base_url=${URLS},api_key=EMPTY,num_concurrent=${CONCURRENCY},adaptive_concurrency=False,adaptive_max_concurrency=128,is_qwen3_vl=True,max_pixels=1003520" \
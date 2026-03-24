#!/bin/bash
set -euo pipefail

# 1. 激活环境
source /mnt/cpfs/yangyicun/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/cpfs/yangyicun/miniconda3/envs/lmms-eval


MODEL="${MODEL:-/mnt/cpfs/yangyicun/data/model/Qwen3-VL-8B-Instruct}"
TASKS="${TASKS:-seedbench,textvqa_val}"
# TASKS="${TASKS:-mmmu,scienceqa,vizwiz_vqa_val,chartqa,ai2d,docvqa_val,ocrbench,seedbench,textvqa_val}"

OUTPUT_PATH="${OUTPUT_PATH:-/mnt/cpfs/yangyicun/eval_result}"
VERBOSITY="${VERBOSITY:-DEBUG}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

# 3. vLLM Server 端口配置
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

echo "[INFO] Checking if vLLM server is running on port ${VLLM_PORT}..."

# 检查 vLLM 接口是否联通的函数
check_vllm() {
    curl -s -o /dev/null -w "%{http_code}" "${VLLM_URL}/models" || echo "000"
}

HTTP_CODE=$(check_vllm)

if [ "$HTTP_CODE" == "200" ]; then
    echo "[INFO] vLLM server is already running."
else
    echo "[INFO] vLLM server not found (HTTP $HTTP_CODE). Starting vLLM server in the background..."
    
    # 核心修改点：使用 setsid 将 vLLM 作为一个完全独立的后台会话启动
    setsid python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --mm-encoder-tp-mode data \
        --trust-remote-code \
        > vllm_server.log 2>&1 &
        
    VLLM_PID=$!
    echo "[INFO] vLLM server started with PID ${VLLM_PID}. Waiting for it to become ready..."
    
    # 轮询等待服务就绪
    MAX_RETRIES=120 # Wait up to 10 minutes (120 * 5s) for large models
    RETRY_COUNT=0
    while [ "$(check_vllm)" != "200" ]; do
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
            echo "[ERROR] vLLM server failed to start within time limit. Check vllm_server.log"
            # 即使启动超时失败，也能安全杀掉遗留进程
            kill ${VLLM_PID} || true
            exit 1
        fi
        echo "[INFO] Waiting for vLLM server... (${RETRY_COUNT}/${MAX_RETRIES})"
    done
    echo "[INFO] vLLM server is ready!"
fi

echo "[INFO] Starting evaluation..."
echo "[INFO] model=${MODEL} tasks=${TASKS}  batch_size=${BATCH_SIZE} tp=${TENSOR_PARALLEL_SIZE}"
echo "[INFO] output_path=${OUTPUT_PATH}"

# 4. 开始评测 (此时按 Ctrl+C 只会杀掉这里的 python 进程，不会影响后台的 vLLM)
python -m lmms_eval \
  --model openai \
  --model_args "model_version=${MODEL},base_url=${VLLM_URL},api_key=EMPTY,num_concurrent=${BATCH_SIZE},adaptive_concurrency=False,adaptive_max_concurrency=1024" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --gen_kwargs "max_new_tokens=4096" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"
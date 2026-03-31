#!/bin/bash
set -euo pipefail

# 1. 自动检测资源
WORLD_SIZE="${WORLD_SIZE:-8}"
TP="${TP:-4}"
NUM_BACKENDS=$(( WORLD_SIZE / TP ))

echo "[INFO] Detected WORLD_SIZE=${WORLD_SIZE}, TP=${TP}. Will start ${NUM_BACKENDS} vLLM instances for Qwen3-VL."

# 2. 激活环境
source /mnt/cpfs/yangyicun/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/cpfs/yangyicun/miniconda3/envs/lmms-eval

MODEL="${MODEL:-/mnt/cpfs/yangyicun/data/model/Qwen3-VL-8B-Instruct}"
TASKS="${TASKS:-seedbench,textvqa_val}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/cpfs/yangyicun/eval_result}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
VERBOSITY="${VERBOSITY:-INFO}" # Options: DEBUG, INFO, WARNING, ERROR

# 3. 循环启动 vLLM 实例
URLS=""
PIDS=()

# Qwen3-VL 使用 8301 起始端口，避免与 Qwen2.5 (8001) 或 Qwen2 (8201) 冲突
for (( i=0; i<$NUM_BACKENDS; i++ )); do
    PORT=$(( 8301 + i ))
    
    # 计算显卡切分 (例如 TP=4, 则 0-3, 4-7)
    START_GPU=$(( i * TP ))
    END_GPU=$(( (i + 1) * TP - 1 ))
    
    # 构建 CUDA_VISIBLE_DEVICES 字符串
    GPUS=""
    for (( g=$START_GPU; g<=$END_GPU; g++ )); do
        GPUS="${GPUS}${g},"
    done
    GPUS=${GPUS%,} # 去掉末尾逗号
    
    LOG_FILE="vllm_instance_qwen3_${PORT}.log"
    
    echo "[INFO] Starting vLLM instance $i on GPUs ${GPUS} at port ${PORT}..."
    
    # 启动 vLLM
    CUDA_VISIBLE_DEVICES=${GPUS} setsid python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --gpu-memory-utilization 0.85 \
        --port "${PORT}" \
        --mm-encoder-tp-mode data \
        --trust-remote-code \
        > "${LOG_FILE}" 2>&1 &
    
    PIDS+=($!)
    URLS="${URLS}http://localhost:${PORT}/v1;"
done

URLS=${URLS%;} # 去掉末尾分号

# 4. 等待所有后端就绪
check_vllm() {
    curl -s -o /dev/null -w "%{http_code}" "$1/models" || echo "000"
}

echo "[INFO] Waiting for all ${NUM_BACKENDS} backends to be ready..."
IFS=';' read -ra URL_ARRAY <<< "$URLS"
for url in "${URL_ARRAY[@]}"; do
    RETRY_COUNT=0
    while [ "$(check_vllm $url)" != "200" ]; do
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ ${RETRY_COUNT} -ge 120 ]; then
            echo "[ERROR] vLLM at $url failed to start."
            exit 1
        fi
        echo "[INFO] Waiting for $url... ($RETRY_COUNT/120)"
    done
done
echo "[INFO] All backends are ready!"

# 5. 执行评测
echo "[INFO] Starting evaluation with Load Balancing..."
echo "[INFO] URLS: ${URLS}"

python -m lmms_eval \
  --model openai \
  --model_args "model_version=${MODEL},base_url=${URLS},api_key=EMPTY,num_concurrent=${BATCH_SIZE},adaptive_concurrency=False,adaptive_max_concurrency=1024" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --gen_kwargs "max_new_tokens=4096" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --verbosity "${VERBOSITY}"

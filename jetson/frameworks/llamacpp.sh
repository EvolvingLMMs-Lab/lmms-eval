# llama.cpp (llama-server in its own container) driven through lmms-eval's OpenAI-compatible backend.
# GGUF weights + f16 vision projector from ggml-org. Latency is measured client side with streaming.
FW_PRECISIONS="q4_k_m q8_0 f16"
LLAMACPP_IMAGE=${LLAMACPP_IMAGE:-llamacpp-jetson:b11135}
LLAMACPP_PORT=${LLAMACPP_PORT:-8090}

_gguf_repo() { echo "$GGUF_REPO"; }
_gguf_file() {
  case "$PRECISION" in
    q4_k_m) echo "$MODEL_TAG-Q4_K_M.gguf" ;;
    q8_0)   echo "$MODEL_TAG-Q8_0.gguf" ;;
    f16)    echo "$MODEL_TAG-f16.gguf" ;;
  esac
}
_mmproj_file() { echo "mmproj-$MODEL_TAG-f16.gguf"; }

fw_assets() {
  echo "model:$(_gguf_repo):$(_gguf_file)"
  echo "model:$(_gguf_repo):$(_mmproj_file)"
}

fw_setup() {
  BACKEND=openai
  MODEL_ARGS="model_version=$MODEL_TAG-$PRECISION,base_url=http://127.0.0.1:$LLAMACPP_PORT/v1,api_key=local,stream_timing=True"
  MODEL_ARGS+=",num_concurrent=1,prefix_aware_queue=False,timeout=600,max_retries=2"
  DOCKER_ARGS+=(--network host)
  LLAMACPP_CONTAINER=llamacpp-bench-$$
}

fw_start() {
  local gguf mmproj
  gguf=$(hf_file "$(_gguf_repo)" "$(_gguf_file)") || return 1
  mmproj=$(hf_file "$(_gguf_repo)" "$(_mmproj_file)") || return 1
  # Image token range matches the other frameworks (IMAGE_MIN/MAX_TOKENS from the model file). Prompt caching is off, as for vLLM:
  # MME asks two questions per image, so a cache would skip the image work on every second sample.
  # cuBLAS accumulates in bf16 instead of fp16: with fp16, one MME image (commonsense_reasoning/0064.png) overflows
  # Qwen2.5-VL-3B and every later request returns "????..." until the server restarts. bf16 keeps fp32's range.
  docker run -d --name "$LLAMACPP_CONTAINER" --runtime nvidia --network host \
    -e GGML_CUDA_CUBLAS_COMPUTE_TYPE="${LLAMACPP_CUBLAS_COMPUTE:-bf16}" \
    -v "$HF_CACHE":"$HF_CACHE":ro "$LLAMACPP_IMAGE" \
    llama-server -m "$gguf" --mmproj "$mmproj" -ngl 999 -c 4096 -np 1 --jinja \
      --image-min-tokens "$IMAGE_MIN_TOKENS" --image-max-tokens "$IMAGE_MAX_TOKENS" --no-cache-prompt --cache-ram 0 \
      --host 127.0.0.1 --port "$LLAMACPP_PORT" >/dev/null
  echo "llama-server: $LLAMACPP_IMAGE $(_gguf_file) (waiting for /health)"
  for _ in $(seq 300); do
    curl -sf "http://127.0.0.1:$LLAMACPP_PORT/health" >/dev/null && return 0
    docker inspect -f '{{.State.Running}}' "$LLAMACPP_CONTAINER" 2>/dev/null | grep -q true || break
    sleep 1
  done
  docker logs "$LLAMACPP_CONTAINER" >"$OUT/server.log" 2>&1 || true
  echo "llama-server failed to start, see $OUT/server.log" >&2
  return 1
}

fw_stop() {
  docker logs "$LLAMACPP_CONTAINER" >"$OUT/server.log" 2>&1 || true
  docker rm -f "$LLAMACPP_CONTAINER" >/dev/null 2>&1 || true
}

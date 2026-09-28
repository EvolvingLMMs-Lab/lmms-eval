# vLLM (in process, lmms-eval `vllm` backend). awq = the model's official AWQ 4-bit checkpoint ($AWQ_REPO).
FW_PRECISIONS="bf16 awq"

_vllm_repo() {
  case "$PRECISION" in
    bf16) echo "$HF_REPO" ;;
    awq)  echo "$AWQ_REPO" ;;
  esac
}

fw_assets() {
  echo "model:$(_vllm_repo)"
}

fw_setup() {
  BACKEND=vllm
  # Fraction of the board's unified memory vLLM may take (weights + KV cache + activations). The default is a fixed
  # budget in GB (13.8, or 18.4 for 7B bf16: 0.45 / 0.6 of the Orin 32GB), so boards with more memory (Thor) keep the
  # same KV-cache budget and memory footprint instead of reserving e.g. 57-74 GB.
  local budget_gb=13.8
  [ "$SIZE" = 7b ] && [ "$PRECISION" = bf16 ] && budget_gb=18.4
  local default_mem
  default_mem=$(awk -v gb="$budget_gb" '/^MemTotal:/ {printf "%.2f", gb * 1048576 / $2}' /proc/meminfo)
  # Same image resolution range as the other frameworks (IMAGE_MIN/MAX_PIXELS from the model file).
  MODEL_ARGS="model=$(hf_snapshot "$(_vllm_repo)"),gpu_memory_utilization=${VLLM_GPU_MEM:-$default_mem},max_model_len=4096,max_pixels=$IMAGE_MAX_PIXELS"
  MODEL_ARGS+=",mm_processor_kwargs={\"min_pixels\":$IMAGE_MIN_PIXELS,\"max_pixels\":$IMAGE_MAX_PIXELS}"
  # No cross-request reuse: MME asks two questions per image, so prefix/image caches would skip most of the
  # vision + prefill work for every second sample, which the HF reference cannot do.
  MODEL_ARGS+=",enable_prefix_caching=False,mm_processor_cache_gb=0"
  # Keep torch.compile / CUDA graph caches between runs (the container is ephemeral).
  DOCKER_ARGS+=(-e VLLM_CACHE_ROOT="$JETSON/.cache/vllm")
}

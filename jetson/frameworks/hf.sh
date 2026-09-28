# Hugging Face transformers (PyTorch, in process) - the reference implementation.
# Sourced by run_eval.sh / download_assets.sh; see run_eval.sh for the variables available here.
FW_PRECISIONS="bf16"

fw_assets() {
  echo "model:$HF_REPO"
}

fw_setup() {
  BACKEND=$HF_BACKEND
  MODEL_ARGS="pretrained=$(hf_snapshot "$HF_REPO"),attn_implementation=${ATTN:-flash_attention_2}"
}

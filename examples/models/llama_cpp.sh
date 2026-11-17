#!/bin/bash
# Local inference via llama.cpp's llama-server, which serves an OpenAI-compatible endpoint
# (https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
# Start the server first; -hf downloads a vision model together with its multimodal projector:
#   llama-server -hf ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --alias qwen2.5-vl-3b --port 8080
# api_key is ignored by llama-server but must be non-empty.

MODEL_VERSION="qwen2.5-vl-3b"  # the --alias given to llama-server

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# base_url points to llama-server; replace localhost:8080 with your server's host and port.
python3 -m lmms_eval \
    --model openai \
    --model_args model_version=$MODEL_VERSION,base_url=http://localhost:8080/v1,api_key=llama-cpp \
    --tasks mme,mmmu_val \
    --batch_size 1

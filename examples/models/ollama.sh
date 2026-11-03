#!/bin/bash
# Local inference via Ollama's OpenAI-compatible endpoint (https://ollama.com).
# Start the server first (`ollama serve`, or the desktop app); api_key is ignored but must be non-empty.

MODEL_VERSION="llava"  # any vision-capable Ollama model, e.g. qwen2.5vl:7b, gemma3
ollama pull $MODEL_VERSION

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# base_url points to the Ollama server; replace localhost:11434 with your Ollama host and port.
python3 -m lmms_eval \
    --model openai \
    --model_args model_version=$MODEL_VERSION,base_url=http://localhost:11434/v1,api_key=ollama,num_concurrent=4 \
    --tasks mme,mmmu_val \
    --batch_size 1

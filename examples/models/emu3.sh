#!/bin/bash
# EMU3 Chat Model Evaluation - standalone file

export HF_HOME="~/.cache/huggingface"

# Define tasks
TASKS="mme,ai2d,vqav2,gqa"

# Get script directory to find helper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Install task-specific dependencies
echo "📦 Installing task-specific dependencies..."
source "${REPO_ROOT}/examples/install_task_deps.sh" "${TASKS}" "${REPO_ROOT}"

# Install Emu3 model dependencies
echo "📦 Installing Emu3 model dependencies..."
cd "${REPO_ROOT}" || exit 1
pip install -e .[emu3]

# Single GPU evaluation
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model emu3 \
    --model_args attn_implementation=flash_attention_2 \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --device cuda:0

# Debug mode: Print first N samples (prompts and answers) to console
# Useful for verifying model behavior and debugging issues
# Example: Print first 10 samples
# accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
#     --model emu3 \
#     --model_args attn_implementation=flash_attention_2,debug_samples=true,num_debug_samples=10 \
#     --tasks mme,ai2d,vqav2,gqa \
#     --batch_size 1 \
#     --device cuda:0


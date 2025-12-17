#!/bin/bash
# EMU3.5 Chat Model Evaluation - standalone file

export HF_HOME="~/.cache/huggingface"

# Define tasks
TASKS="mme,ai2d,vqav2,gqa"

# Get script directory to find helper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Install task-specific dependencies
echo "📦 Installing task-specific dependencies..."
source "${REPO_ROOT}/examples/install_task_deps.sh" "${TASKS}" "${REPO_ROOT}"

# Install Emu3.5 model dependencies
echo "📦 Installing Emu3.5 model dependencies..."
cd "${REPO_ROOT}" || exit 1
pip install -e .[emu3_5]

# Single GPU evaluation
accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model emu3p5 \
    --model_args attn_implementation=flash_attention_2 \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --device cuda:0

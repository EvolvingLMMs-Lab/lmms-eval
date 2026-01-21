#!/bin/bash
# Download UniWorld-V1 model from HuggingFace

echo "======================================"
echo "Downloading UniWorld-V1 Model"
echo "======================================"

# Model path
MODEL_NAME="LanguageBind/UniWorld-V1"
LOCAL_DIR="./models/UniWorld-V1"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install -U huggingface-hub
fi

echo ""
echo "Downloading model to: ${LOCAL_DIR}"
echo "This may take a while..."
echo ""

# Download model
huggingface-cli download ${MODEL_NAME} \
    --local-dir ${LOCAL_DIR} \
    --local-dir-use-symlinks False

echo ""
echo "======================================"
echo "Download completed!"
echo "Model saved to: ${LOCAL_DIR}"
echo "======================================"
echo ""
echo "Now you can run:"
echo "  bash uniworld_general.sh \"2\" \"chartqa100\" \"./logs/chartqa\" \"${LOCAL_DIR}\""

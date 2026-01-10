#!/bin/bash

# Test the evaluation server using Python-based CICD launcher
# This allows for better GPU management and test orchestration

# Default values
MODEL_NAME=""
GPU_COUNT=8
VERBOSE="--verbose"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --no-verbose)
            VERBOSE=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-name NAME     Model to test (e.g., qwen2_5_vl)"
            echo "                        Default: (empty - tests all models)"
            echo "  --gpu-count NUM       Number of GPUs to use (default: 8)"
            echo "  --no-verbose          Disable verbose output"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting evaluation CICD tests..."
echo "Model: ${MODEL_NAME:-all models}"
echo "GPU Count: $GPU_COUNT"

# Get the absolute path of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Build the command
CMD="python $REPO_DIR/test/eval/run_cicd.py $VERBOSE --gpu-count $GPU_COUNT"

# Add model name if specified
if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi

# Run the Python-based test launcher
echo "Running: $CMD"
$CMD

EXIT_CODE=$?

echo "Evaluation CICD tests completed with exit code: $EXIT_CODE"
exit $EXIT_CODE


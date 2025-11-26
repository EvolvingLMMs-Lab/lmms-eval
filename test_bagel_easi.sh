#!/bin/bash
# Test Bagel model on EASI spatial reasoning benchmarks
# Usage: bash test_bagel_easi.sh

set -e  # Exit on error

echo "=========================================="
echo "Testing Bagel on EASI Spatial Benchmarks"
echo "=========================================="

# Configuration
MODEL_NAME="bagel"
MODEL_PATH="${MODEL_PATH:-/path/to/BAGEL-7B-MoT}"  # Set this to your model path
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/bagel_easi_results}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-}"  # Leave empty for full test, or set to a number for quick test

# EASI benchmarks
BENCHMARKS=(
    "mmsibench"        # MMSI
    "embspatialbench"  # OmniSpatial
    "mindcubebench"    # MindCube
    "spatial457"       # SpatialViz
)

# Check if model path is set
if [ "$MODEL_PATH" = "/path/to/BAGEL-7B-MoT" ]; then
    echo "ERROR: Please set MODEL_PATH environment variable"
    echo "Example: export MODEL_PATH=/path/to/your/BAGEL-7B-MoT"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Model Path: $MODEL_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Output Directory: $OUTPUT_DIR"
if [ -n "$LIMIT" ]; then
    echo "  Limit: $LIMIT samples per task"
fi
echo ""

# Function to run evaluation
run_eval() {
    local task=$1
    local log_file="$OUTPUT_DIR/${task}_$(date +%Y%m%d_%H%M%S).log"

    echo "--------------------------------------"
    echo "Running: $task"
    echo "Log: $log_file"
    echo "--------------------------------------"

    # Build command
    CMD="python -m lmms_eval \
        --model $MODEL_NAME \
        --model_args pretrained=$MODEL_PATH \
        --tasks $task \
        --batch_size $BATCH_SIZE \
        --device $DEVICE \
        --output_path $OUTPUT_DIR"

    # Add limit if specified
    if [ -n "$LIMIT" ]; then
        CMD="$CMD --limit $LIMIT"
    fi

    echo "Command: $CMD"
    echo ""

    # Run evaluation
    eval $CMD 2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $task completed successfully"
    else
        echo "✗ $task failed"
        return 1
    fi
    echo ""
}

# Run all benchmarks
FAILED_TASKS=()
for task in "${BENCHMARKS[@]}"; do
    if ! run_eval "$task"; then
        FAILED_TASKS+=("$task")
    fi
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total tasks: ${#BENCHMARKS[@]}"
echo "Completed: $((${#BENCHMARKS[@]} - ${#FAILED_TASKS[@]}))"
echo "Failed: ${#FAILED_TASKS[@]}"

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tasks:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    exit 1
else
    echo ""
    echo "✓ All tasks completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
fi

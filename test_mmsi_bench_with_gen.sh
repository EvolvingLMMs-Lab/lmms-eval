#!/bin/bash

# Test script for MMSI-Bench generation subtasks (Ovis-U1) with Visual CoT
# Output directory: /home/yifeishen/blob/mount/xiaoyu/ovis_u1/with_gen/

set -e  # Exit on error

# Configuration
OUTPUT_DIR="/blob/mount/xiaoyu/ovis_u1/with_gen"
MODEL="ovis_u1"
MODEL_ARGS="pretrained=AIDC-AI/Ovis-U1-3B"
BATCH_SIZE=1
DEVICE="cuda:0"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Define MMSI-Bench subtasks with visual CoT (generation)
TASKS=(
    "mmsi_attribute_appr_visual_cot"
    "mmsi_attribute_meas_visual_cot"
    "mmsi_motion_obj_visual_cot"
    "mmsi_motion_cam_visual_cot"
    "mmsi_msr_visual_cot"
)

# Print configuration
echo "=========================================="
echo "MMSI-Bench With-Generation Test Script"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Device: ${DEVICE}"
echo "Tasks: ${#TASKS[@]}"
echo "=========================================="
echo ""

# Run each task
for task in "${TASKS[@]}"; do
    echo "----------------------------------------"
    echo "Running task: ${task}"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"

    # Create task-specific output directory
    TASK_OUTPUT="${OUTPUT_DIR}/${task}"
    mkdir -p "${TASK_OUTPUT}"

    # Run the evaluation
    uv run python -m lmms_eval \
        --model "${MODEL}" \
        --model_args "${MODEL_ARGS}" \
        --tasks "${task}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --output_path "${TASK_OUTPUT}" \
        2>&1 | tee "${TASK_OUTPUT}/run.log"

    echo "Task ${task} completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Keep GPU occupied to prevent reclamation
echo ""
echo "=========================================="
echo "Starting GPU occupation to prevent reclamation..."
echo "Press Ctrl+C to stop GPU occupation"
echo "=========================================="

uv run python lmms_eval/occupy_gpu.py --device 0 --memory 20

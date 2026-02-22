#!/bin/bash

# Qwen3-VL Evaluation Script with SGLang Backend
# This script demonstrates how to evaluate Qwen3-VL models using SGLang for accelerated inference
#
# Requirements:
# - sglang>=0.4.6
# - qwen-vl-utils
# - CUDA-enabled GPU(s)
#
# Installation:
# uv add "sglang[all]" qwen-vl-utils
# OR
# pip install "sglang[all]>=0.4.6" qwen-vl-utils

# ============================================================================
# Configuration
# ============================================================================

# Model Configuration
# Available Qwen3-VL models:
# - Qwen/Qwen3-VL-30B-A3B-Instruct
# - Qwen/Qwen3-VL-30B-A3B-Thinking
# - Qwen/Qwen3-VL-235B-A22B-Instruct
# - Qwen/Qwen3-VL-235B-A22B-Thinking
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"

# Parallelization Settings
# Adjust based on your GPU configuration
TENSOR_PARALLEL_SIZE=4  # Number of GPUs for tensor parallelism (tp_size in SGLang)

# Memory and Performance Settings
GPU_MEMORY_UTILIZATION=0.85  # mem_fraction_static in SGLang (0.0 - 1.0)
BATCH_SIZE=64                # Batch size for evaluation

# SGLang Specific Settings
MAX_PIXELS=1605632           # Maximum pixels for image processing
MIN_PIXELS=784               # Minimum pixels (28x28)
MAX_FRAME_NUM=32            # Maximum number of video frames
THREADS=16                  # Number of threads for decoding visuals

# Task Configuration
# Common tasks: mmmu_val, mme, mathvista, ai2d, etc.
TASKS="mmmu_val,mme"

# Output Configuration
OUTPUT_PATH="./logs/qwen3vl_sglang"
LOG_SAMPLES=true
LOG_SUFFIX="qwen3vl_sglang"

# Evaluation Limits (optional)
# LIMIT=100  # Uncomment to limit number of samples (for testing)

# ============================================================================
# Environment Configuration
# ============================================================================
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ============================================================================
# EXAMPLE 1: Basic SGLang Usage (Without MCP Tools)
# ============================================================================
# This is the standard evaluation without tool calling support.
# The model will process image/video queries and return responses directly.
#
# Key Parameters:
# - model: The model identifier
# - tensor_parallel_size: Number of GPUs for tensor parallelism
# - gpu_memory_utilization: GPU memory fraction to use
# - max_pixels/min_pixels: Image resolution constraints
# - max_frame_num: Maximum frames for video processing
# - threads: Thread count for visual processing

echo "=========================================="
echo "Qwen3-VL Evaluation with SGLang"
echo "=========================================="
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Pixels: $MAX_PIXELS"
echo "Output Path: $OUTPUT_PATH"
echo "=========================================="

# Build the command
CMD="uv run python -m lmms_eval \
    --model sglang \
    --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_frame_num=${MAX_FRAME_NUM},threads=${THREADS} \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}"

# Add optional arguments
if [ "$LOG_SAMPLES" = true ]; then
    CMD="$CMD --log_samples --log_samples_suffix ${LOG_SUFFIX}"
fi

if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit ${LIMIT}"
fi

# Execute
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================="

# ============================================================================
# EXAMPLE 2: SGLang with MCP Client Tools (Tool-Enabled Evaluation)
# ============================================================================
# This example demonstrates how to enable MCP (Model Context Protocol) client
# for tool calling support with SGLang.
#
# IMPORTANT: Before running this, you need to:
# 1. Create an MCP server that exposes tools (e.g., image processing, web search)
# 2. The MCP server should be a Python script that implements tool definitions
# 3. Pass the path to the MCP server script via mcp_server_path parameter
#
# How MCP Tool Calling Works with SGLang:
# ─────────────────────────────────────────
# 1. User sends a request with a question
# 2. SGLang processes the message and generates text
# 3. The function_call_parser detects if tool calls are in the generated text
#    (finish_reason == "tool_calls")
# 4. If tool calls are detected:
#    a. Parse the tool call function name and arguments from generated text
#    b. Retrieve tool definition from MCPClient
#    c. Execute the tool via MCPClient.run_tool(tool_name, arguments)
#    d. Convert tool result to OpenAI-compatible format
#    e. Append tool result to conversation as {"role": "tool", ...}
#    f. Generate next response with updated context (max_turn times)
# 5. Continue until model produces final text or max_turn is reached
#
# Tool Calling Loop in Code (from sglang.py):
# ──────────────────────────────────────────────
# while keep_rolling and turn_count < max_turn:
#     output = await self.client.async_generate(...)
#     if function_call_parser.has_tool_call(output["text"]):
#         tool_calls = function_call_parser.parse_non_stream(output["text"])
#         for tool_call in tool_calls:
#             result = await self.mcp_client.run_tool(tool_call.name, args)
#             # Convert result to OpenAI format
#             tool_messages.append({"role": "tool", "name": tool_call.name, "content": result})
#         messages.append(assistant_response)
#         messages.extend(tool_messages)
#         # Prepare next input for model with tool results
#         turn_count += 1
#
# Example with MCP tools enabled:
# (Uncomment the following lines to use with MCP server)
#
# # Path to MCP server implementation
# MCP_SERVER_PATH="/path/to/mcp_server.py"
# WORK_DIR="/tmp/sglang_mcp_work"
#
# CMD="uv run python -m lmms_eval \
#     --model sglang \
#     --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_frame_num=${MAX_FRAME_NUM},threads=${THREADS},mcp_server_path=${MCP_SERVER_PATH},work_dir=${WORK_DIR},max_turn=5 \
#     --tasks ${TASKS} \
#     --batch_size 1 \
#     --output_path ${OUTPUT_PATH}_with_mcp \
#     --log_samples --log_samples_suffix ${LOG_SUFFIX}_mcp"
#
# eval $CMD

# ============================================================================
# Parameter Reference
# ============================================================================
# model                    : Model identifier (required)
# tensor_parallel_size     : Number of GPUs for tensor parallelism (default: 1)
# gpu_memory_utilization   : GPU memory fraction (0.0-1.0, default: 0.8)
# batch_size               : Batch size for evaluation (default: 1)
# max_pixels               : Max image resolution (default: 1605632)
# min_pixels               : Min image resolution (default: 28*28=784)
# max_frame_num            : Max frames for videos (default: 768)
# fps                      : Frames per second for video sampling (optional)
# nframes                  : Fixed number of frames for video (default: 32)
# threads                  : Thread count for visual processing (default: 16)
# mcp_server_path          : Path to MCP server script for tool calling (optional)
# work_dir                 : Working directory for MCP tools (default: /tmp/...)
# max_turn                 : Maximum tool calling turns (default: 5)
# chat_template            : Custom chat template jinja file (optional)
# json_model_override_args : JSON args to override model config (optional)
#
#
# ============================================================================
# Tool Calling Best Practices
# ============================================================================
# 1. TOOL DESIGN:
#    - Keep tools focused on single tasks
#    - Provide clear, specific descriptions
#    - Define input schema with required fields
#    - Return results in structured format
#
# 2. MCP SERVER:
#    - Must be a standalone Python script
#    - Should handle errors gracefully
#    - Return results in TextContent or ImageContent format
#    - Avoid long-running operations (timeouts)
#
# 3. CONFIGURATION:
#    - Set appropriate max_turn value (5-10 recommended)
#    - Use batch_size=1 when tools are enabled (sequential processing)
#    - Allocate sufficient work_dir space for temporary files
#    - Monitor GPU memory with tool execution
#
# 4. DEBUGGING:
#    - Use --verbosity DEBUG to see tool call details
#    - Check work_dir for saved images/videos
#    - Validate MCP server responds correctly: 
#      `python mcp_server.py` should start without errors
#    - Test tool functions independently before evaluation
#
# ============================================================================

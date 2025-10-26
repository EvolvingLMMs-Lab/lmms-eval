#!/bin/bash

# ============================================================================
# Async OpenAI Compatible Model Example
# ============================================================================
# This script demonstrates how to use the async_openai_compatible model with:
# 1. Basic video/image evaluation (without MCP tools)
# 2. Tool-enabled evaluation (with MCP client)
#
# The AsyncOpenAIChat class supports asynchronous processing of requests
# using OpenAI-compatible API servers (e.g., vLLM, local LLMs with OpenAI wrapper)
# ============================================================================

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY='EMPTY'

MODEL_VERSION="Qwen/Qwen3-VL-4B-Instruct"

# ============================================================================
# EXAMPLE 1: Basic Usage (Without MCP Tools)
# ============================================================================
# This is the simplest usage pattern without tool calling.
# The model will process video/image queries and return responses.
#
# Key Parameters:
# - model_version: The model name (used in API calls)
# - max_pixels: Maximum pixels for image resolution (default: 151200)
# - base_url: OpenAI API base URL
# - api_key: API key (use 'EMPTY' for local servers)
# - num_cpus: Number of concurrent workers (controls parallelism)
# - timeout: Request timeout in seconds
# - is_qwen3_vl: Set to True for Qwen3-VL specific formatting, set to False for other models

accelerate launch --num_processes=1 --main_process_port 12345 -m lmms_eval \
    --model async_openai \
    --model_args model_version=$MODEL_VERSION,max_pixels=151200,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=8,timeout=6000,is_qwen3_vl=True \
    --tasks videomme \
    --batch_size 1 \
    --output_path ./logs/ \
    --log_samples --verbosity DEBUG

# ============================================================================
# EXAMPLE 2: With MCP Client Tools (Tool-Enabled Evaluation)
# ============================================================================
# This example demonstrates how to enable MCP (Model Context Protocol) client
# for tool calling. The model can now use external tools during inference.
#
# IMPORTANT: Before running this, you need to:
# 1. Create an MCP server that exposes tools (e.g., image processing, web search)
# 2. The MCP server should be a Python script that implements tool definitions
# 3. Pass the path to the MCP server script via mcp_server_path parameter
#
# How MCP Tool Calling Works:
# ─────────────────────────────
# 1. User sends a request with a question
# 2. The model receives the message and processes it
# 3. The OpenAI API may decide to call a tool (finish_reason == "tool_calls")
# 4. The MCPClient retrieves tool definitions from the MCP server
# 5. The model calls the tool via MCPClient.run_tool()
# 6. The tool result is converted to OpenAI format
# 7. The result is sent back to the model for continuation
# 8. Steps 3-7 repeat in a loop until the model produces final text output
#
# Tool Calling Loop in Code (from async_openai.py):
# ──────────────────────────────────────────────────
# while response.choices[0].finish_reason == "tool_calls":
#     for tool_call in response.choices[0].message.tool_calls:
#         result = await self.mcp_client.run_tool(call.function.name, args)
#         # Convert result to OpenAI format
#         tool_messages.append({"role": "tool", "content": result})
#     # Send tool results back to model for next iteration
#     response = await self.client.chat.completions.create(
#         model=model_version,
#         messages=messages + tool_messages,
#         tools=tool_definitions,
#         tool_choice="auto"
#     )

# Example with MCP tools enabled:
# (Uncomment the following lines to use)
#
# accelerate launch --num_processes=1 --main_process_port 12345 -m lmms_eval \
#     --model async_openai \
#     --model_args model_version=$MODEL_VERSION,max_pixels=151200,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=8,timeout=6000,mcp_server_path=/path/to/mcp_server.py,work_dir=/tmp/mcp_work \
#     --tasks videomme \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --log_samples --verbosity DEBUG

# ============================================================================
# Parameter Reference
# ============================================================================
# model_version          : Model name for API calls (required)
# base_url               : OpenAI API endpoint (required)
# api_key                : API key (required, use 'EMPTY' for local servers)
# num_cpus               : Number of concurrent async workers (default: cpu_count//2)
# timeout                : Request timeout in seconds (default: 600)
# max_retries            : Number of retries on failure (default: 5)
# max_pixels             : Max image resolution (default: 151200)
# min_pixels             : Min image resolution (default: 28*28)
# max_frames             : Max frames for videos (default: 768)
# fps                    : Frames per second for video sampling (optional)
# nframes                : Fixed number of frames for video (default: 64)
# is_qwen3_vl            : Enable Qwen3-VL specific formatting (default: False)
# mcp_server_path        : Path to MCP server script for tool calling (optional)
# work_dir               : Working directory for MCP tools (default: /tmp/...)
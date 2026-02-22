# LongVT Evaluation Tasks

This directory contains evaluation tasks for [LongVT](https://github.com/EvolvingLMMs-Lab/LongVT), an agentic framework for long video understanding via native tool calling.

## Available Tasks

| Task | Description |
|------|-------------|
| `longvt_non_think` | Direct answer mode |
| `longvt_reasoning` | Thinking mode with `<think>` and `<answer>` tags |
| `longvt_tool` | Tool calling mode with MCP server |

## Dataset

The benchmark uses [VideoSIAH-Eval](https://huggingface.co/datasets/longvideotool/VideoSIAH-Eval) from Hugging Face.

## Usage

### Environment Variables

```bash
# Environment variables
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_MODEL_NAME="judge"
export OPENAI_BASE_URL="http://your-judge-server-ip:8000/v1"
export OPENAI_API_KEY="EMPTY"
export USE_LLM_JUDGE=True
export DECORD_EOF_RETRY_MAX=409600

# Cache enabled (optional)
# export LMMS_EVAL_USE_CACHE=True
# export LMMS_EVAL_HOME="your_cache_directory"
```

### Arguments

```bash
CKPT_PATH=$1                # Path to model checkpoint
TASK_NAME=$2                # Evaluation task name (longvt_non_think / longvt_reasoning / longvt_tool)
IS_QWEN3_VL=$3              # Whether using Qwen3-VL model (True/False)
MAX_FRAME_NUM=${4:-768}     # Number of frames (Default: 768)

# Path to MCP server for tool calling (only needed for longvt_tool)
MCP_PATH="examples/mcp_server/crop_video_mcp_server.py"
```

---

### 1. Direct Answer / Thinking Mode (`longvt_non_think`, `longvt_reasoning`)

```bash
accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
    --model async_openai \
    --model_args model_version=$CKPT_PATH,fps=1,max_frames=$MAX_FRAME_NUM,max_pixels=50176,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=1,timeout=12000,is_qwen3_vl=$IS_QWEN3_VL \
    --tasks $TASK_NAME \
    --batch_size 1 \
    --output_path ./eval_logs \
    --log_samples
```

---

### 2. Tool Calling (`longvt_tool`)

#### Step 1: Start MCP Server

```bash
# Qwen3-VL does not need additional chat template
if [ "$IS_QWEN3_VL" == "False" ]; then
    vllm serve $CKPT_PATH \
        --chat-template examples/chat_templates/tool_call_qwen2_5_vl.jinja \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        --data-parallel-size 8 \
        --trust-remote-code &
else
    vllm serve $CKPT_PATH \
        --tool-call-parser hermes \
        --enable-auto-tool-choice \
        --data-parallel-size 8 \
        --trust-remote-code &
fi
sleep 240  # Wait for vLLM server to be ready
```

#### Step 2: Run Evaluation

```bash
accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
    --model async_openai \
    --model_args model_version=$CKPT_PATH,mcp_server_path=$MCP_PATH,fps=1,max_frames=$MAX_FRAME_NUM,max_pixels=50176,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=1,timeout=12000,is_qwen3_vl=$IS_QWEN3_VL \
    --tasks $TASK_NAME \
    --batch_size 1 \
    --output_path ./eval_logs \
    --log_samples
```

## Citation

```bibtex
@article{yang2025longvt,
  title={LongVT: Incentivizing "Thinking with Long Videos" via Native Tool Calling},
  author={Yang, Zuhao and Wang, Sudong and Zhang, Kaichen and Wu, Keming and Leng, Sicong and Zhang, Yifan and Li, Bo and Qin, Chengwei and Lu, Shijian and Li, Xingxuan and Bing, Lidong},
  journal={arXiv preprint arXiv:2511.20785},
  year={2025}
}
```

## References

- [LongVT Repo](https://github.com/EvolvingLMMs-Lab/LongVT)
- [LongVT Paper](https://arxiv.org/abs/2511.20785)

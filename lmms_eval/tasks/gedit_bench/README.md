# GEdit-Bench

## Requirements

### 1. VLM Judge Server

This benchmark requires a Vision-Language Model (VLM) to judge the quality of edited images using VIEScore. We use **Qwen2.5-VL-72B-Instruct** by default.

You need to serve the VLM using vLLM or any OpenAI-compatible API server

### 2. Environment Variables

Set the following environment variables before running the evaluation:

| Variable | Required | Description |
|----------|----------|-------------|
| `VIESCORE_API_KEY` | Yes | API key for the VLM server (use `"EMPTY"` for local vLLM) |
| `VIESCORE_API_BASE` | Yes | Base URL of the VLM server (e.g., `http://localhost:8000/v1`) |
| `VIESCORE_MODEL_NAME` | Yes | Model name/path for evaluation |

Example:
```bash
export VIESCORE_API_KEY="EMPTY"
export VIESCORE_API_BASE="http://localhost:8000/v1"
export VIESCORE_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
```

### 3. Required Arguments for Image Editing Models

When evaluating image editing models, you **must** use the following model arguments for optimal editing performance:

```bash
--model_args ...,num_timesteps=50,cfg_img_scale=2.0,cfg_renorm_type="text_channel",cfg_interval=0.0
```

### 4. Process with Media Flag

You **must** include the `--process_with_media` flag to ensure images are available in the processing documents:

```bash
--process_with_media
```

## Usage Example

```bash
export VIESCORE_API_KEY="EMPTY"
export VIESCORE_API_BASE="http://localhost:8000/v1"
export VIESCORE_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

accelerate launch --num_processes=1 -m lmms_eval \
    --model bagel_lmms_engine \
    --model_args pretrained=your_model_path,device_map=cuda,output_image_dir=./logs/images,num_timesteps=50,cfg_img_scale=2.0,cfg_renorm_type="text_channel",cfg_interval=0.0 \
    --tasks gedit_bench \
    --batch_size 1 \
    --output_path ./logs/ \
    --log_samples \
    --process_with_media
```


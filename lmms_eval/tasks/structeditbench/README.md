# StructEditBench (Structured-Visuals) Benchmark

This task evaluates **image editing models** with a 2-stage pipeline (following `qwen_scoring.py`):

- Stage 1 (**Vision-QA**): ask questions on the **edited image** and get a short answer.
- Stage 2 (**Judge**): a text-only judge decides **Correct/Incorrect** given (Question, GT, Model Response).

Metrics:
- `structeditbench_editing_accuracy` (%)
- `structeditbench_maintain_accuracy` (%)
- `structeditbench_weighted_accuracy` (%) = \(0.9 \cdot editing + 0.1 \cdot maintain\)

## Requirements

You need an **OpenAI-compatible** API server that supports **multimodal** chat completions (e.g., vLLM serving Qwen2.5-VL).

## Environment Variables (ImgEdit-style)

Set these before running:

| Variable | Required | Description |
|----------|----------|-------------|
| `STRUCTEDITBENCH_API_KEY` | Yes | API key for the eval server (use `"EMPTY"` for local vLLM) |
| `STRUCTEDITBENCH_BASE_URL` | Yes | Base URL (e.g., `http://localhost:8000/v1`) |
| `STRUCTEDITBENCH_EVAL_MODEL_NAME` | Yes | Model name/path used for **both** QA and judge (unless override judge model) |
| `STRUCTEDITBENCH_JUDGE_MODEL_NAME` | No | Optional separate model for stage-2 judging |
| `STRUCTEDITBENCH_TIMEOUT` | No | API timeout seconds (default: `180`) |
| `STRUCTEDITBENCH_MAX_RETRIES` | No | Retries on transient errors (default: `3`) |
| `STRUCTEDITBENCH_CALL_DELAY` | No | Delay between API calls in seconds (default: `0.5`) |
| `STRUCTEDITBENCH_MAX_QA` | No | Cap QA list length for quick tests |

Example:

```bash
export STRUCTEDITBENCH_API_KEY="EMPTY"
export STRUCTEDITBENCH_BASE_URL="http://localhost:8000/v1"
export STRUCTEDITBENCH_EVAL_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
```

## Required Runtime Flag

If your model produces edited images and the task needs to load them, include:

```bash
--process_with_media
```

## Usage Example

```bash
export STRUCTEDITBENCH_API_KEY="EMPTY"
export STRUCTEDITBENCH_BASE_URL="http://localhost:8000/v1"
export STRUCTEDITBENCH_EVAL_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

accelerate launch --num_processes=1 -m lmms_eval \
  --model bagel_lmms_engine \
  --model_args pretrained=your_model_path,device_map=cuda,output_image_dir=./logs/structeditbench_images \
  --tasks structeditbench \
  --batch_size 1 \
  --output_path ./logs/ \
  --log_samples \
  --process_with_media
```



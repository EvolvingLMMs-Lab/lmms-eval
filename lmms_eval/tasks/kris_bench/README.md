# KRIS-Bench

KRIS-Bench is an image editing benchmark that evaluates:
- **Consistency** (non-instructed content preserved)
- **Instruction following** (requested edit is performed correctly)
- **Quality** (visual quality / artifacts)
- **Knowledge plausibility** (only for knowledge categories)

This lmms-eval task:
1. Generates an image per sample (conditioned on 1+ source images).
2. Scores the generated image with an **OpenAI-compatible multimodal judge** inside `process_results`.

## Data Source

This task loads data from Hugging Face Hub:
- `wukeming11/kris-bench` (images embedded in dataset)

Images are loaded automatically via `--process_with_media` flag. No local paths needed!

## Judge Server

You must provide an OpenAI-compatible API server that supports multimodal chat completions (e.g., vLLM serving Qwen2.5-VL).

Required:

```bash
export KRIS_BENCH_API_KEY="EMPTY"
export KRIS_BENCH_BASE_URL="http://localhost:8000/v1"
export KRIS_BENCH_EVAL_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
```

Optional:
- `KRIS_BENCH_JUDGE_MODEL_NAME`: use a separate model for judging
- `KRIS_BENCH_TIMEOUT` (default: 180)
- `KRIS_BENCH_MAX_RETRIES` (default: 3)
- `KRIS_BENCH_CALL_DELAY` (default: 0.5)

## Important Runtime Flag

**Required**: `--process_with_media` flag to load images from HF dataset and edited images from model output.

## Usage Example

```bash
export KRIS_BENCH_API_KEY="EMPTY"
export KRIS_BENCH_BASE_URL="http://localhost:8000/v1"
export KRIS_BENCH_EVAL_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

accelerate launch --num_processes=8 -m lmms_eval \
    --model bagel_lmms_engine \
    --model_args pretrained=your_model_path,device_map=cuda,output_image_dir=./logs/kris_bench_images \
    --tasks kris_bench \
    --batch_size 1 \
    --output_path ./logs/ \
    --log_samples \
    --process_with_media
```

## How It Works

1. **Input images**: Loaded from HF dataset `doc["ori_images"]` via `--process_with_media`
2. **Model generation**: Model saves edited images to `output_image_dir`
3. **Evaluation**: Task reads edited images from `pred["images"][0]` path, sends to VLM judge
4. **Scoring**: Judge returns consistency, instruction, quality (and knowledge) scores

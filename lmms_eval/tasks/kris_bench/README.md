# KRIS-Bench

KRIS-Bench is an image editing benchmark that evaluates:
- **Consistency** (non-instructed content preserved)
- **Instruction following** (requested edit is performed correctly)
- **Quality** (visual quality / artifacts)
- **Knowledge plausibility** (only for knowledge categories)

This lmms-eval task:
1. Generates an image per sample (conditioned on 1+ source images).
2. Scores the generated image with an **OpenAI-compatible multimodal judge** inside `process_results`.

## Data

This task loads a local jsonl index:

- `lmms_eval/tasks/kris_bench/kris_bench.jsonl`

Images are loaded at runtime from:

- `$KRIS_BENCH_DATA_ROOT/{category}/{filename}`

You can generate the jsonl index with:

```bash
python lmms_eval/tasks/kris_bench/prepare_dataset.py \
  --bench_root /path/to/KRIS_Bench \
  --output_jsonl lmms_eval/tasks/kris_bench/kris_bench.jsonl
```

## Judge Server (ImgEdit-style env vars)

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

If your model outputs edited images and the task needs to load them, include:

```bash
--process_with_media
```

## Usage Example

```bash
export KRIS_BENCH_DATA_ROOT="/path/to/KRIS_Bench/KRIS_Bench"
export KRIS_BENCH_API_KEY="EMPTY"
export KRIS_BENCH_BASE_URL="http://localhost:8000/v1"
export KRIS_BENCH_EVAL_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

accelerate launch --num_processes=1 -m lmms_eval \
  --model bagel_lmms_engine \
  --model_args pretrained=your_model_path,device_map=cuda,output_image_dir=./logs/kris_bench_images \
  --tasks kris_bench \
  --batch_size 1 \
  --output_path ./logs/ \
  --log_samples \
  --process_with_media
```



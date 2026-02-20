# MotionBench

MotionBench integration for lmms-eval video motion understanding evaluation.

- Source benchmark: https://motion-bench.github.io/
- Metadata source: `video_info.meta.jsonl` from the upstream MotionBench repository
- Task family: `motionbench` (group) with fullset task `motionbench_full`

## Available task names

- `motionbench` (group)
- `motionbench_full`

## Smoke run (API backend)

```bash
python -m lmms_eval \
  --model openai \
  --model_args model=gpt-4o-mini \
  --tasks motionbench_full \
  --limit 5 \
  --batch_size 1
```

If local videos are available, set:

```bash
export MOTIONBENCH_VIDEO_DIR=/path/to/motionbench/videos
```

By default, lmms-eval will also auto-download missing videos from `zai-org/MotionBench` into Hugging Face cache when needed.

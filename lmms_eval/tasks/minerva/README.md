# MINERVA task in lmms-eval

This task supports local video files with a YouTube URL fallback.

## Metadata source

`minerva.yaml` reads metadata from:

- `https://huggingface.co/datasets/lmms-lab-eval/minerva/resolve/main/minerva.json`

## Video resolution priority

`minerva_doc_to_visual` resolves videos in this order:

1. Local files via `MINERVA_VIDEO_DIR`
2. Fallback YouTube URL reconstruction (`https://www.youtube.com/watch?v=<video_id>`)

## Local video mode

Set:

```bash
export MINERVA_VIDEO_DIR="/absolute/path/to/videos"
```

Expected filenames: `<video_id>.mp4` / `.webm` / `.mkv` / `.mov`.

## Dummy model evaluation for video-read simulation

Use `dummy` to simulate request flow and local video reads without real model/API inference.

```bash
uv run python -m lmms_eval \
  --model dummy \
  --model_args "read_bytes=65536,response=A,allow_remote=false,fail_on_missing=true" \
  --tasks minerva \
  --batch_size 1 \
  --limit 50 \
  --output_path ./logs/minerva_dummy \
  --verbosity INFO
```

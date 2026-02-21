# MINERVA task in lmms-eval

This task supports both local video files and Lance-backed video blobs.

## Metadata source

`minerva.yaml` reads metadata from:

- `https://huggingface.co/datasets/lmms-lab-eval/minerva/resolve/main/minerva.json`

## Video resolution priority

`minerva_doc_to_visual` resolves videos in this order:

1. Local files via `MINERVA_VIDEO_DIR`
2. Lance blobs via `MINERVA_LANCE_VIDEO_URI`
3. Fallback YouTube URL reconstruction

## Local video mode

Set:

```bash
export MINERVA_VIDEO_DIR="/absolute/path/to/videos"
```

Expected filenames: `<video_id>.mp4` / `.webm` / `.mkv` / `.mov`.

## Lance mode

Set:

```bash
export MINERVA_LANCE_VIDEO_URI="hf://datasets/lmms-lab-eval/minerva/data/train.lance"
export MINERVA_LANCE_VIDEO_ID_COLUMN="video_id"
export MINERVA_LANCE_VIDEO_BLOB_COLUMN="video_blob"
export MINERVA_LANCE_CACHE_DIR="~/.cache/lmms_eval/minerva_lance_videos"
```

Install optional dependencies:

```bash
uv add pylance pyarrow
```

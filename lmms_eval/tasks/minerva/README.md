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

# Optional Lance runtime tuning for remote object storage
export LANCE_IO_THREADS="64"
export LANCE_CPU_THREADS="8"
```

Install optional dependencies:

```bash
uv add pylance pyarrow
```

Note: this is the Python package `pylance` that exposes module name `lance` at runtime.

## Benchmark resolver latency

Use the benchmark tool to report absolute latency for `minerva_doc_to_visual`:

```bash
uv run python tools/bench_minerva_video_resolution.py \
  --metadata-json data/minerva/minerva.json \
  --mode lance \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 200 \
  --sample-unique-video
```

Output includes `startup_ms`, plus separate `cold_*` and `warm_*` latency stats (`mean_ms`, `p50_ms`, `p95_ms`) for direct before/after comparisons.

## Benchmark pipeline latency (raw vs Lance storage)

Use the pipeline benchmark to compare end-to-end latency under the same model decode path:

```bash
uv run python tools/bench_minerva_pipeline_latency.py \
  --local-video-dir /absolute/path/to/minerva/videos \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 100 \
  --batch-size 1 \
  --decode-num-frames 8
```

This reports local and Lance mode metrics with the same pipeline decode implementation and prints comparison ratios.

Interpretation guidance:

- This benchmark keeps the decode path fixed and changes only storage mode (local raw files vs Lance blob-backed resolution).
- In a local pre-downloaded setup, pipeline latency is typically close between modes because decode dominates total time.
- Lance is usually more valuable for data packaging, reproducibility, and remote/object-storage workflows than for single-node decode speedups.

When Lance is likely to show stronger practical gains:

- Remote object storage / Hub-based datasets where many small-file operations are expensive.
- Multi-machine or CI-style evaluation where a single versioned Lance package improves reproducibility.
- Repeated subset runs (`video_id` selection) with warm cache behavior across runs.


## Dummy model evaluation for video-read simulation

Use `dummy_video_reader` to simulate request flow and local video reads without real model/API inference.

```bash
uv run --with pylance --with pyarrow python -m lmms_eval \
  --model dummy_video_reader \
  --model_args "read_bytes=65536,response=A,allow_remote=false,fail_on_missing=true" \
  --tasks minerva \
  --batch_size 1 \
  --limit 50 \
  --output_path ./logs/minerva_dummy_video_reader \
  --verbosity INFO
```

For Lance mode, set `MINERVA_LANCE_VIDEO_URI` and related env vars first.

## Build Lance dataset with blob-oriented defaults

```bash
uv run --with pylance --with pyarrow python tools/minerva_to_lance.py \
  --metadata-json data/minerva/minerva.json \
  --videos-dir data/minerva/videos \
  --output data/minerva_hf_package/data/train.lance \
  --batch-size 8 \
  --mode overwrite \
  --max-rows-per-file 512 \
  --max-rows-per-group 64 \
  --max-bytes-per-file-gb 4 \
  --data-storage-version stable
```

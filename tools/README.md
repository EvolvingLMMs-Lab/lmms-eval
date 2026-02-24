# Tools

Utility scripts for lmms-eval development and dataset preparation.

## Scripts

### `get_split_zip.py`

Split large ZIP files into smaller parts for hosting services with file size limits.

```bash
# Split into 5GB parts (default)
python tools/get_split_zip.py dataset.zip ./output/

# Split into 2GB parts
python tools/get_split_zip.py dataset.zip ./output/ --max-size 2GB
```

### `regression.py`

Run regression tests across git branches to compare model performance.

```bash
python tools/regression.py --tasks ocrbench,mmmu_val --limit 8
```

### `minerva_to_lance.py`

Build MINERVA Lance dataset from metadata and local videos.

```bash
python tools/minerva_to_lance.py \
  --metadata-json data/minerva/minerva.json \
  --videos-dir data/minerva/videos \
  --output data/minerva_hf_package/data/train.lance \
  --mode overwrite
```

### `bench_minerva_video_resolution.py`

Benchmark absolute latency of MINERVA video resolution (`minerva_doc_to_visual`).

```bash
python tools/bench_minerva_video_resolution.py \
  --metadata-json data/minerva/minerva.json \
  --mode lance \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 200 \
  --sample-unique-video
```

### `bench_minerva_pipeline_latency.py`

Benchmark end-to-end MINERVA pipeline latency for local raw storage vs Lance storage using the same decode path.

```bash
python tools/bench_minerva_pipeline_latency.py \
  --local-video-dir /absolute/path/to/minerva/videos \
  --lance-uri hf://datasets/lmms-lab-eval/minerva/data/train.lance \
  --limit 100 \
  --batch-size 1 \
  --decode-num-frames 8
```

This benchmark keeps decode behavior fixed and compares only storage mode. On local disks it is often near-parity; Lance tends to help more in remote/object-storage and reproducible multi-machine setups.

### `bench_video_decode_backends.py`

Benchmark video decode backend latency and throughput on the same local video file across `pyav`, `torchcodec`, and `dali`.

```bash
python tools/bench_video_decode_backends.py \
  --video /absolute/path/to/video.mp4 \
  --backends pyav,torchcodec,dali \
  --num-frames 8 \
  --iterations 30 \
  --output /tmp/video_decode_backends.json
```

`pyav` is the default reference. If optional backends are not installed, the script records backend-specific errors in the JSON report without failing the whole run.

## Notebooks

### `make_image_hf_dataset.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EvolvingLMMs-Lab/lmms-eval/blob/main/tools/make_image_hf_dataset.ipynb)

Tutorial for creating properly formatted Hugging Face datasets with images. Demonstrates the complete workflow from raw data to HF Hub upload.

---

## Archived Modules

The following modules were removed during cleanup. They can be found in the `main` branch if needed:

- **`lite/`** - LMMs-Eval Lite for dataset core-set selection using embedding-based sampling
- **`live_bench/`** - Separate package for LiveBench data generation and website screenshot capture

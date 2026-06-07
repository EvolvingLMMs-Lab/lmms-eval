# VSTAT

[VSTAT](https://github.com/vision-x-nyu/vstat) is a video benchmark for visual
state tracking. It evaluates whether multimodal models can track fine-grained
state changes, counts, locations, attributes, and temporal order across
long-form videos.

## Dataset

The annotations and redistributable videos are hosted on
[nyu-visionx/vstat](https://huggingface.co/datasets/nyu-visionx/vstat).
Synthetic and self-recorded videos are included in the Hugging Face dataset.
YouTube videos are not redistributed and must be downloaded with the scripts
bundled in the dataset repository.

```bash
pip install -U yt-dlp
# Install ffmpeg with your system package manager if it is not already available.

huggingface-cli download nyu-visionx/vstat \
  --repo-type=dataset \
  --local-dir /path/to/vstat

cd /path/to/vstat
python scripts/download_youtube.py --resolution-map youtube_resolutions.json
bash scripts/redact.sh
```

The official benchmark numbers use the downloaded clips after the redaction
step. Point lmms-eval at the prepared dataset:

```bash
export VSTAT_VIDEO_ROOT=/path/to/vstat
```

The task downloads `vstat_qa_clean.json` from Hugging Face by default. For
offline runs, or if you want to force the local copy from the prepared dataset,
also set:

```bash
export VSTAT_QA_PATH=/path/to/vstat_qa_clean.json
```

## Evaluation

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=784,max_pixels=50176,max_num_frames=128 \
  --tasks vstat \
  --batch_size 1 \
  --output_path ./logs/vstat \
  --log_samples
```

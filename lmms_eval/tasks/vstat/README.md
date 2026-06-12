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

On first use, the task downloads the Hugging Face dataset into `$HF_HOME/vstat`,
matching the cache directory pattern used by other video tasks. If YouTube clips
are missing, the task also runs the bundled YouTube downloader and redaction
script from that cache directory. Existing files are skipped on later runs.

```bash
pip install -U yt-dlp
# Install ffmpeg with your system package manager if it is not already available.
```

Set `HF_HOME` if you want the cache somewhere other than `~/.cache/huggingface`.

For offline runs, or to use a separately prepared dataset root, set:

```bash
export VSTAT_VIDEO_ROOT=/path/to/vstat
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

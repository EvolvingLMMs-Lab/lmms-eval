# LLaVA-OneVision-2-8B-Instruct · Reproduction Guide

Minimal commands to reproduce every benchmark we report for
`lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` (verified on
8 × A100-80GB).

> This is a fork of [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)
> tailored for LLaVA-OneVision-2 reproduction. The upstream README is
> preserved at [`README_upstream.md`](./README_upstream.md).

---

## 1. Environment

```bash
docker build -t lmms-eval-ov2:latest -f dockerfile/Dockerfile .

docker run --privileged --gpus all --ipc=host --shm-size=16g --network=host \
  -v $(pwd):/workspace/lmms-eval \
  -v /path/to/hf_cache:/hf_cache \
  -e HF_HOME=/hf_cache \
  -it lmms-eval-ov2:latest /bin/bash

# Inside the container
cd /workspace/lmms-eval
pip install -e . --no-deps

# Required for `video_backend=codec` (installs the `codec-video-prep-legacy-exact` CLI).
# Use this exact pinned legacy-exact build — do NOT install the regular
# `codec-video-prep` package on PyPI.
python3 -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    codec-video-prep-legacy-exact==0.2.5.post2
```

---

## 2. Reproduction commands

We provide two thin launcher scripts — one per video backend — and a
setting table per benchmark. Pick the row, set the env vars, run the
launcher.

### 2.1 Frames backend (uniform sampling)

Launcher: `examples/llava_onevision2_repro/run_frames.sh` — required env: `TASK`, `F`
(number of frames), `MP` (`min_pixels` = `max_pixels`). Set `IL=1` for
interleaved-subtitle tasks (auto-exports
`LMMS_IL_NOPREFIX=1 LMMS_IL_FILTER_NOISE=1`).

```bash
TASK=<task> F=<num_frames> MP=<max_pixels> [IL=1] \
    bash examples/llava_onevision2_repro/run_frames.sh
```

| Benchmark (`TASK`)                       | `F`  | `MP`     | `IL` | Score             |
|------------------------------------------|-----:|---------:|:---:|------------------:|
| `ov2_videomme_short_wo_sutitle`          |  128 | 321 489 |  –  | 81.56 (perception) |
| `ov2_videomme_medium_wo_sutitle`         |  256 | 136 900 |  –  | 72.56 (perception) |
| `ov2_videomme_long_wo_sutitle`           |  640 | 102 400 |  –  | 62.33 (perception) |
| `videomme_short_interleaved_subtitle`    |  128 | 233 289 |  1  | 83.33 (perception) |
| `videomme_medium_interleaved_subtitle`   |  448 | 128 164 |  1  | 78.00 (perception) |
| `videomme_long_interleaved_subtitle`     |  640 |  84 100 |  1  | 69.22 (perception) |
| `lvbench`                                |  768 |  84 100 |  –  | 55.46 |
| `mlvu_dev`                               |  512 |  72 900 |  –  | 76.62 (perception) |
| `videoeval_pro`                          |  768 |  84 100 |  –  | 61.45 (overall) |
| `vsibench`                               |  128 | 153 664 |  –  | 70.94 (overall) |
| `timelens_activitynet`                   |  128 | 153 664 |  –  | mIOU 53.75 |
| `timelens_charades`                      |  128 | 153 664 |  –  | mIOU 53.49 |
| `timelens_qvhighlights`                  |  128 | 153 664 |  –  | mIOU 66.43 |
| `videommev2_interleaved_subtitle`        |   64 | 330 000 |  1  | 18.34 |

Example:

```bash
TASK=mlvu_dev F=512 MP=72900 bash examples/llava_onevision2_repro/run_frames.sh
TASK=videomme_long_interleaved_subtitle F=640 MP=84100 IL=1 \
    bash examples/llava_onevision2_repro/run_frames.sh
```

### 2.2 Codec backend (canvas-packed video tokens)

Launcher: `examples/llava_onevision2_repro/run_codec.sh` — required env: `TASK`, `TC`
(`codec_target_canvas` and `max_num_frames`). Optional: `TS` (timestamp
decimals, default `1`; set `2` for sub-second timestamp tasks like
JumpScore), `IL=1` for interleaved-subtitle tasks.

`min_pixels=100352` and `max_pixels=313600` are hard-coded as the codec
defaults; override via `MIN_PX` / `MAX_PX` if needed.

```bash
TASK=<task> TC=<num_canvases> [TS=2] [IL=1] \
    bash examples/llava_onevision2_repro/run_codec.sh
```

| Benchmark (`TASK`)                  | `TC` | `TS` | `IL` | Score      |
|-------------------------------------|-----:|:----:|:----:|-----------:|
| `videommev2_interleaved_subtitle`   |  64  |  1   |  1   | 19.89      |
| `JumpScore`                         | 128  |  2   |  –   | mAP 0.7549 |
| `timelens_activitynet`              |  64  |  1   |  –   | mIOU 51.23 |
| `timelens_charades`                 |  64  |  1   |  –   | mIOU 50.09 |
| `timelens_qvhighlights`             |  64  |  1   |  –   | mIOU 63.53 |

Example:

```bash
TASK=JumpScore TC=128 TS=2 bash examples/llava_onevision2_repro/run_codec.sh

TASK=videommev2_interleaved_subtitle TC=64 IL=1 \
    bash examples/llava_onevision2_repro/run_codec.sh
```

> By default the launcher generates canvases online into
> `out/<task>_codec_*/online_codec_*`. To reuse pre-generated assets,
> point `LLAVA_CODEC_OFFLINE_ROOT` at a directory whose layout matches
> `<video_stem>__<sha1_8>/` (see `_find_offline_asset_dir` in
> `lmms_eval/models/chat/llava_onevision2.py` for the asset-generation
> command).

> The codec offline-asset directory layout is `<video_stem>__<sha1_8>/`
> with `canvas_*.jpg`, `src_patch_position.npy`, `frame_ids.npy`,
> `meta.json`, `_DONE`. See `_find_offline_asset_dir` in
> `lmms_eval/models/chat/llava_onevision2.py` for the asset-generation
> command.

---

## 3. Notes

- All scripts log to `out/<task>_<config>/run.log`; final
  results land in `*_results.json` in the same directory.
- For VideoMME no-subtitle / IL, LVBench, MLVU-dev, VideoEval-Pro,
  `min_pixels = max_pixels = r²` (square / equal-aspect token budget).
  `MP = 153 664` corresponds to a fixed 392×392 resolution.
- The frames-backend `(F, MP)` settings were selected from a full
  `(resolution, num_frames)` grid sweep over each benchmark.

---

## 4. Reference

- Model: <https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct>
- Bundled processor commit: <https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct/commit/5a75eaf7>

# VBench

This integration generates scorer-compatible videos for the official VBench
and VBench 2.0 prompt suites. It deliberately separates generation from
scoring: the lmms-eval metric named `generated` is only the successful-video
ratio; final VBench quality and semantic scores must be computed with the
official VBench scorer.

## Hugging Face dataset

All tasks load [`pufanyi/VBench`](https://huggingface.co/datasets/pufanyi/VBench)
at the immutable revision
`f95c61c6ea5e45800d69ae0c9fe3824449d79548`. The conversion is built from
VBench revision `1ee42dada7a2f7cfaf4290e8a02d087f6f8ee425` and retains the
unmodified source registries and Apache-2.0 license.

The dataset has combined configs and one config per official dimension:

| Config | Requests | Protocol |
|---|---:|---|
| `vbench` | 6,220 | 5 videos per prompt; 25 for `temporal_flickering` |
| `vbench2` | 3,209 | 3 videos per prompt; 20 for `Diversity` |

Exact duplicate prompt strings are merged in the combined configs. Every row
contains its sample index and a deterministic, non-cherry-picked seed. The
official Wan2.2 submission did not publish its per-video seeds, so these seeds
make a new lmms-eval run reproducible without claiming bit-for-bit identity
with the submitted videos.

To rebuild the dataset:

```bash
uv run --no-project --with datasets --with huggingface-hub \
  python lmms_eval/tasks/vbench/build_dataset.py \
  --output-dir /tmp/vbench-hf
```

Add `--repo-id OWNER/VBench --push-to-hub` to publish it.

## Generate Wan2.2 videos

Install the project and video-generation dependencies with uv:

```bash
uv venv --python 3.11
uv pip install -e '.[video]'
uv pip install 'diffusers>=0.35.2' imageio imageio-ffmpeg
```

The `wan2_2_t2v` defaults match the native Wan2.2 720p recipe at the exact
native commit cited by VBench: 1280×720, 81 frames, 40 denoising steps,
guidance 4/3, flow shift 12, the official negative prompt, and 16 fps. The
converted Diffusers weights are also pinned to revision
`5be7df9619b54f4e2667b2755bc6a756675b5cd7` by default. VBench did not publish
its complete launch command or per-video seeds, so this is the closest public
recipe rather than a claim of bit-for-bit generation identity.

```bash
torchrun --nproc-per-node=8 -m lmms_eval \
  --model wan2_2_t2v \
  --model_args pretrained=Wan-AI/Wan2.2-T2V-A14B-Diffusers,revision=5be7df9619b54f4e2667b2755bc6a756675b5cd7,output_dir=/data/wan22-vbench \
  --tasks vbench \
  --batch_size 1 \
  --log_samples
```

Use a new `output_dir` when changing generation settings. `vbench` is the
preferred combined task because it generates each unique prompt/sample only
once. The `vbench_dimensions` group remains available for dimension-specific
runs. VBench 2.0 follows the same pattern with `vbench2` and
`vbench2_dimensions`.

Generated filenames follow the scorer contract:

- VBench: `<output_dir>/vbench/<prompt>-<sample_index>.mp4`
- VBench 2.0 combined: `<output_dir>/vbench2/<prompt[:180]>-<sample_index>.mp4`
- VBench 2.0 per dimension:
  `<output_dir>/vbench2/<Official_Dimension>/<prompt[:180]>-<sample_index>.mp4`

## Score and compare

VBench first samples 25 videos for every `temporal_flickering` prompt, then
uses its RAFT-based static filter to select and rename five videos. Run that
official preprocessing before scoring a newly generated `vbench` directory.
Other dimensions can be scored directly from the unfiltered directory; score
`temporal_flickering` from the filter output and merge the resulting dimension
JSON entries. Preserve the filter manifest: upstream traverses the directory
without sorting and accepts the first five clips that pass its threshold, so
re-running it over a differently ordered directory can select another subset.

Use the scorer at the same pinned VBench revision as the prompt registry:

```bash
uv venv --python 3.10 /tmp/vbench-venv
uv pip install --python /tmp/vbench-venv/bin/python \
  --index https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
uv pip install --python /tmp/vbench-venv/bin/python \
  --no-build-isolation \
  'git+https://github.com/Vchitect/VBench@1ee42dada7a2f7cfaf4290e8a02d087f6f8ee425'
```

Follow the official VBench `static_filter` and `evaluate` commands for the
dimension runs. Once all 16 entries are in one `*_eval_results.json`, compare
raw dimensions and leaderboard aggregates with:

```bash
python lmms_eval/tasks/vbench/compare_wan22_results.py \
  /data/results/wan22_eval_results.json \
  --baseline no_prompt \
  --output-json /data/results/wan22_comparison.json
```

The two [official leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
baselines encoded in the comparison tool are:

| Submission | Total | Quality | Semantic |
|---|---:|---:|---:|
| Wan2.2, no prompt extension | 82.61 | 85.03 | 72.92 |
| Wan2.2, Qwen prompt extension | 84.23 | 85.42 | 79.50 |

### Public-sample reproduction

As a scorer-level control, we re-scored all 375 `temporal_flickering`
videos in the official no-prompt-extension archive (75 prompts × 5 filtered
samples). Every selected ZIP member passed its CRC check, and all videos
decoded as 1280×720, 81 frames, 16 fps. The calculation is the same
adjacent-frame MAE implemented by VBench v0.1.5 at the pinned revision.

| Source | Official | Re-scored | Delta (percentage points) |
|---|---:|---:|---:|
| Wan2.2 public samples, `temporal_flickering` | 98.92 | 97.9832 | -0.9368 |

This is not a rounding difference. It also is not a comparison against newly
generated lmms-eval videos: it isolates the public artifact and scorer. Given
that the formula is unchanged, the mismatch indicates that the leaderboard
submission and the later public archive, or an unreported evaluation
condition, are not sufficient to reproduce this dimension exactly. The
upstream request for Wan2.2 reproduction details remains open in
[`Vchitect/VBench#202`](https://github.com/Vchitect/VBench/issues/202).

The compact result and provenance are checked in as
[`wan22_official_sample_reproduction.json`](wan22_official_sample_reproduction.json).
After extracting the archive, reproduce the detailed result with:

```bash
uv run --no-project --with 'numpy<2' --with 'opencv-python-headless<5' \
  python lmms_eval/tasks/vbench/reproduce_wan22_temporal_flickering.py \
  --videos-dir /data/Wan2.2-T2V-A14B/videos \
  --output-json /data/results/wan22_temporal_flickering.json
```

A full 16-dimension scorer rerun additionally requires the complete 46.44-GiB
archive and all VBench detector checkpoints. The lightweight control above is
the completed measured comparison included with this change; the comparison
tool is ready to consume a future full `*_eval_results.json` without changing
the official baseline.

The exact Qwen-expanded prompts used in the second submission were not
published. It is therefore possible to re-score the official released videos
exactly and to generate a new deterministic Qwen-extended run, but not to
reconstruct the official Qwen submission byte for byte from public artifacts.

## Official sample archive

The no-prompt-extension release is a 12-part split ZIP containing 4,720 MP4
files (944 unique prompts × 5 filtered samples), approximately 46.44 GiB. The
downloader validates every segment against its Drive metadata size, preventing
an interrupted response from being mistaken for a complete file:

```bash
uv run --no-project --with 'gdown>=5.2,<6' \
  python lmms_eval/tasks/vbench/download_official_wan22.py \
  --output-dir /data/Wan2.2-T2V-A14B-wo-prompt-extend

uv run --no-project --with 'gdown>=5.2,<6' \
  python lmms_eval/tasks/vbench/download_official_wan22.py \
  --output-dir /data/Wan2.2-T2V-A14B-wo-prompt-extend \
  --verify-only
```

After verification, merge and extract without deleting the source volumes:

```bash
cd /data/Wan2.2-T2V-A14B-wo-prompt-extend
zip -s 0 Wan2.2-T2V-A14B.zip --out Wan2.2-T2V-A14B-Full.zip
unzip Wan2.2-T2V-A14B-Full.zip -d videos
```

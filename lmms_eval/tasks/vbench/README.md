# VBench

This integration generates scorer-compatible videos for the official VBench
and VBench 2.0 prompt suites. It deliberately separates generation from
scoring: the lmms-eval metric named `generated` is only the successful-video
ratio; final VBench quality and semantic scores must be computed with the
official VBench scorer.

## Hugging Face dataset

All tasks load [`pufanyi/VBench`](https://huggingface.co/datasets/pufanyi/VBench).
The conversion is built from VBench revision
`1ee42dada7a2f7cfaf4290e8a02d087f6f8ee425` and retains the unmodified source
registries and Apache-2.0 license.

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

## Generate videos

Install the project and video-generation dependencies with uv:

```bash
uv venv --python 3.11
uv pip install -e '.[video]'
uv pip install 'diffusers>=0.35.2' imageio imageio-ffmpeg
```

The tasks are model-agnostic and can be used with any lmms-eval video generator.
For example, to generate VBench videos with the Diffusers Wan2.2 backend:

```bash
torchrun --nproc-per-node=8 -m lmms_eval \
  --model wan2_2_t2v \
  --model_args pretrained=Wan-AI/Wan2.2-T2V-A14B-Diffusers,output_dir=/data/vbench-videos \
  --tasks vbench \
  --batch_size 1 \
  --log_samples
```

Generation settings belong in `--model_args`; the task does not override model
defaults. Use a new `output_dir` when changing them. `vbench` is the preferred
combined task because it generates each unique prompt/sample only once. The
`vbench_dimensions` group remains available for dimension-specific runs.
VBench 2.0 follows the same pattern with `vbench2` and `vbench2_dimensions`.

Generated filenames follow the scorer contract:

- VBench: `<output_dir>/vbench/<prompt>-<sample_index>.mp4`
- VBench 2.0 combined: `<output_dir>/vbench2/<prompt[:180]>-<sample_index>.mp4`
- VBench 2.0 per dimension:
  `<output_dir>/vbench2/<Official_Dimension>/<prompt[:180]>-<sample_index>.mp4`

## Score

VBench first samples 25 videos for every `temporal_flickering` prompt, then
uses its RAFT-based static filter to select and rename five videos. Run that
official preprocessing before scoring a newly generated `vbench` directory.
Other dimensions can be scored directly from the unfiltered directory; score
`temporal_flickering` from the filter output and merge the resulting dimension
JSON entries. Preserve the filter manifest: upstream traverses the directory
without sorting and accepts the first five clips that pass its threshold, so
re-running it over a differently ordered directory can select another subset.

Use the official scorer at the same VBench source revision as the converted
prompt registry:

```bash
uv venv --python 3.10 /tmp/vbench-venv
uv pip install --python /tmp/vbench-venv/bin/python \
  --index https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
uv pip install --python /tmp/vbench-venv/bin/python \
  --no-build-isolation \
  'git+https://github.com/Vchitect/VBench@1ee42dada7a2f7cfaf4290e8a02d087f6f8ee425'
```

Follow the upstream VBench `static_filter` and `evaluate` instructions for the
dimension runs. lmms-eval's `generated` metric only reports artifact-generation
success; the VBench quality and semantic scores come from the official scorer.

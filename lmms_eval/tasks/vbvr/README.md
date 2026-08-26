# VBVR-Bench

Image-to-video reasoning benchmark ([Video-Reason/VBVR-EvalKit](https://github.com/Video-Reason/VBVR-EvalKit)).
Given a first-frame image and a textual instruction, the model generates an
MP4 video; scoring is rule-based and per-task (no LLM judge, no CLIP).

- **500 samples** = 100 tasks × 5 instances each
- **In-Domain_50** (250 samples) — tasks overlapping with the training set
- **Out-of-Domain_50** (250 samples) — held-out tasks
- **5 cognitive categories**: Abstraction, Knowledge, Perception, Spatiality, Transformation

## Tasks

| Task name             | Scope                            |
| --------------------- | -------------------------------- |
| `vbvr`                | Full 500-sample bench            |
| `vbvr_in_domain`      | In-Domain_50 only                |
| `vbvr_out_of_domain`  | Out-of-Domain_50 only            |

## Data Cache

The HF dataset card (`Video-Reason/VBVR-Bench-Data`) carries the base64-encoded
first-frame plus **relative** paths to `ground_truth.mp4`, `first_frame.png`,
`final_frame.png`, `prompt.txt` etc. The task config uses
`dataset_kwargs.cache_dir: vbvr`, so lmms-eval downloads the dataset snapshot
and links it under `$HF_HOME/vbvr` by default. The rule-based evaluators resolve
GT files from that cache path automatically.

If you already have a local checkout, you can still override the GT root with:

```bash
export VBVR_GT_PATH=/data/VBVR-Bench
```

## Running

```bash
python -m lmms_eval \
  --model fastvideo \
  --model_args model=Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --tasks vbvr \
  --batch_size 1 \
  --log_samples \
  --output_path logs
```

The model must output JSON of the form:

```json
{"text": "", "videos": ["/abs/path/to/generated.mp4"]}
```

### SGLang Diffusion (Wan2.2-I2V)

Recent SGLang releases include a native Diffusion runtime for Wan. Its current
Diffusion extra requires PyAV 16 while lmms-eval's base environment pins an
older PyAV, so use a dedicated uv environment and install the two projects
sequentially. The command below pins the build validated on this cluster; newer
Linux wheels require a newer glibc than its current 2.31 runtime. Keep lmms-eval
to one process; SGLang launches and manages the configured GPU workers itself.

```bash
uv venv .venv-sglang --python 3.12
uv pip install --python .venv-sglang/bin/python -e .
uv pip install --python .venv-sglang/bin/python --prerelease=allow "sglang[diffusion]==0.5.10.post1"

.venv-sglang/bin/python -m lmms_eval \
  --model sglang \
  --model_args "model=Wan-AI/Wan2.2-I2V-A14B-Diffusers,runtime=diffusion,num_gpus=4,enable_cfg_parallel=true,ulysses_degree=2,text_encoder_cpu_offload=true,pin_cpu_memory=true" \
  --tasks vbvr \
  --batch_size 1 \
  --log_samples \
  --output_path logs/sglang_wan22_vbvr
```

`runtime=diffusion` is optional for Wan repository names and local Diffusers
checkpoints with a `model_index.json`; `--model sglang` detects those
automatically. Use `--model sglang_diffusion` to select the same adapter
explicitly. A complete configurable launcher is available at
`examples/models/sglang_wan22_vbvr.sh`.

### Full example (multi-GPU Wan2.2-I2V)

```fish
#!/usr/bin/env fish
# Run from the lmms-eval repo root.
cd /path/to/lmms-eval; or exit 1

# Rule-based VBVR scorers read the GT mp4s/pngs from this root.
# By default this is populated automatically at $HF_HOME/vbvr.
# Uncomment this only if you want to use an existing local checkout.
# set -gx VBVR_GT_PATH /path/to/VBVR-Bench

set MODEL_DIR   /path/to/Wan2.2-I2V-A14B-Diffusers

set MODEL_ARGS "model=$MODEL_DIR"
set MODEL_ARGS "$MODEL_ARGS,data_parallel=4,num_gpus=2,sp_size=2,tp_size=1"
set MODEL_ARGS "$MODEL_ARGS,num_inference_steps=50,num_frames=81"
set MODEL_ARGS "$MODEL_ARGS,height=1024,width=1024,fps=16"
set MODEL_ARGS "$MODEL_ARGS,dit_cpu_offload=False,text_encoder_cpu_offload=True"
set MODEL_ARGS "$MODEL_ARGS,image_encoder_cpu_offload=False,vae_cpu_offload=False"
set MODEL_ARGS "$MODEL_ARGS,enable_torch_compile=True"

exec stdbuf -oL -eL .venv/bin/python -m lmms_eval eval \
    --model fastvideo \
    --model_args $MODEL_ARGS \
    --tasks vbvr \
    --batch_size 1 \
    --log_samples \
    --output_path logs
```

FastVideo-generated videos land in `$HF_HOME/lmms_eval/generated_videos/fastvideo`
by default; SGLang-generated videos land under `logs/sglang_diffusion` unless
`output_dir` is set in `--model_args`. Per-sample logs and aggregated metrics land under `--output_path`, and
the detailed VBVR evaluation JSON is written through `generate_submission_file()`
under `--output_path/submissions/`. Add `--use_cache <path>` only if you want
lmms-eval response caching in addition to FastVideo's generated-mp4 reuse. Tune
`data_parallel`, `num_gpus`, `sp_size`, and the `*_cpu_offload` flags to match
your hardware.

## Metrics

- `vbvr_overall` — sample-weighted mean across both splits
- `vbvr_in_domain`, `vbvr_out_of_domain`
- `vbvr_{abstraction,knowledge,perception,spatiality,transformation}` — per-category means

All metrics are continuous in `[0, 1]` — higher is better.

## Vendored code

`vbvr_bench/` is copied from upstream commit of
[Video-Reason/VBVR-EvalKit](https://github.com/Video-Reason/VBVR-EvalKit)
(Apache 2.0, see `vbvr_bench/LICENSE`). Rule-based scorers depend only on
`numpy`, `opencv-python`, `Pillow`, `imageio`, `tqdm`.

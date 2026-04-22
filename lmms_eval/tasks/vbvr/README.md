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

## One-time setup

The HF dataset card (`Video-Reason/VBVR-Bench-Data`) carries the base64-encoded
first-frame plus **relative** paths to `ground_truth.mp4`, `first_frame.png`,
`final_frame.png`, `prompt.txt` etc. The rule-based evaluators read those GT
files, so you must first download the repo and point `VBVR_GT_PATH` at it:

```bash
hf download Video-Reason/VBVR-Bench-Data \
  --repo-type dataset \
  --local-dir /data/VBVR-Bench

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

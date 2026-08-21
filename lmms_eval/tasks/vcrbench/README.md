# VCRBench

This task ports the official [VCRBench](https://github.com/pritamqu/VCRBench) protocol
("VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models").
Each of the 365 examples shows one procedural activity whose clips were shuffled and
concatenated into a single video; every clip is introduced by a burnt-in `Clip N` title card.
The model must recover the causal order of the clips, so this is free-form generation and not
multiple choice. The prompt is the official one and asks for `Correct order: <clip numbers>`.

The dataset is public. Media is downloaded from
https://huggingface.co/datasets/pritamqu/VCRBench and linked under `$HF_HOME/vcrbench`, so the
videos live at `$HF_HOME/vcrbench/videos/video_<N>.mp4`. Existing media can instead be exposed
through `VCRBENCH_VIDEO_DIR` (a directory holding the `video_<N>.mp4` files) or `VCRBENCH_ROOT`
(a directory holding a `videos/` subdirectory).

```bash
accelerate launch -m lmms_eval --model <model> --tasks vcrbench --batch_size 1
```

## Metrics

| Metric | Official name | Definition |
| --- | --- | --- |
| `vcrbench_accuracy` | `avg_accuracy` | Exact match of the whole predicted clip order. |
| `vcrbench_step_accuracy` | `avg_step_accuracy` | Position-wise match over all clips; a length mismatch scores zeros. |
| `vcrbench_weighted_accuracy` | `weighted_avg_accuracy` | Unweighted mean of the exact-match accuracy over the 12 goal classes. |

Answer parsing follows the official `process.py`: the response is split into sentences, the
first sentence whose comma-separated fields form a permutation of `1..N` wins, and a response
with no such sentence is scored as `N` zeros. Reasoning models are supported as a pre-step:
`<think>` blocks are stripped, and an `<answer>...</answer>` or `\boxed{...}` wrapper is read
before the official cascade runs.

## Sanity targets

The official random baseline is 7.95 `avg_accuracy` / 24.26 `avg_step_accuracy`. A model that
scores near those numbers is not reading the title cards.

The paper evaluates Qwen2.5-VL with `fps=1`, `max_pixels=360*420` (151200) and
`max_new_tokens=512`; pass the frame and pixel settings through the model arguments, for
example `--model_args ...,max_pixels=151200,fps=1`.

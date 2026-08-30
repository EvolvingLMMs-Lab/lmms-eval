# LongVideo-Reason

This task ports the official [LongVideo-Reason-eval](https://github.com/NVlabs/Long-RL)
protocol from "Scaling RL to Long Videos" ([arXiv:2507.07966](https://arxiv.org/abs/2507.07966)),
the benchmark introduced alongside LongVILA-R1. It is a balanced set of 1,000 long-video
questions across four reasoning perspectives: **temporal** (294), **goal and purpose** (288),
**plot and narrative** (283) and **spatial** (135). Each example is a four-option
multiple-choice question whose options are already embedded in the question text.

```bash
accelerate launch -m lmms_eval --model <model> --tasks longvideo_reason --batch_size 1
```

## Getting the data

The annotations and the videos live in **two different repositories**, and only the
annotations download automatically.

* Annotations — [`LongVideo-Reason/longvideo-reason`](https://huggingface.co/datasets/LongVideo-Reason/longvideo-reason),
  `test` split, 1,000 rows.
* Videos — [`LongVideo-Reason/longvideo_eval_videos`](https://huggingface.co/datasets/LongVideo-Reason/longvideo_eval_videos),
  ten `longvideo_eval_subset<N>.tar.gz` shards totalling **195 GB**.

Extract every shard into one parent directory so that a `longvila_videos/` directory results,
then point the task at the **parent**:

```bash
export LONGVIDEO_REASON_VIDEO_DIR=/path/to/parent   # contains longvila_videos/
```

`LONGVIDEO_REASON_ROOT` and the usual `LMMS_EVAL_MEDIA_ROOT` / `$HF_HOME/longvideo_reason`
fallbacks also work. The split ships **867 `.mp4`, 130 `.webm` and 3 `.mkv`** files, so a
decoding stack that only handles MP4 silently loses 13% of the benchmark.

## Metrics

| Metric | Definition |
| --- | --- |
| `longvideo_reason_overall_accuracy` | Headline accuracy over all 1,000 rows, using the layered extractor below. |
| `longvideo_reason_strict_accuracy` | Accuracy under the official byte-exact string comparison. Reproduces the paper's number. |
| `longvideo_reason_wellformed_accuracy` | Accuracy over the 995 rows whose option block is intact (see below). |
| `longvideo_reason_format_accuracy` | Share of completions matching `<think>...</think>\s*<answer>...</answer>`, the official `format_reward`. |
| `longvideo_reason_{temporal,goal,spatial,plot}_accuracy` | Per-perspective accuracy, the four-way breakdown the paper reports. |

The official reference implementation (`longvideo-reason/eval.py`) computes only overall
accuracy and format accuracy; the per-perspective split is reported in the paper but not
computed there, so this task derives it from the `problem_type` field.

### Answer extraction

Two numbers are reported on purpose, because the official comparison is strict enough that
small formatting differences dominate it.

`longvideo_reason_strict_accuracy` reproduces `accuracy_reward()` exactly: the `<answer>...
</answer>` span of the completion is compared to the `<answer>...</answer>` span of the gold
with `==`, after the official `"Therefore the final answer is: "` special case. No
normalisation. A completion of `<answer>B.</answer>` against a gold of `<answer>B</answer>`
scores **zero** under this metric, and that is upstream behaviour, not a bug to repair.

`longvideo_reason_overall_accuracy` runs the official extraction first and only falls back
when it does not yield an offered option letter: `\boxed{X}`, then the shared
`extract_mcq_answer` cascade. Every layer is additive, so a completion the official parser
already resolves is returned unchanged and `overall >= strict` holds by construction.
Letters the example does not actually offer are always rejected.

The official implementation additionally tries `math_verify.parse`/`verify` before its regex
path. On a single-letter multiple-choice answer that symbolic path adds nothing the regex
path does not already cover, so it is omitted here.

## Reading the score

> **The answer key is not uniform. Read the accuracy against 44.1%, not 25%.**
> The gold distribution over the 1,000 test rows is **B = 441, C = 299, A = 153, D = 107**.
> A model that always answers `B` scores 44.1%. A model scoring near 44% has not
> necessarily learned anything, and a model below it is worse than a constant.

> **Five rows (0.5%) carry a malformed option block.** In `problem_id` 147, 743 and 825 the
> data generator leaked its own deliberation into the options
> (`"A. But this requires the test-taker to weigh..."`), and `problem_id` 379 and 857 ship
> with **no options at all** and cannot be answered as multiple choice by any model. They are
> deliberately **kept**, because upstream scores all 1,000 and changing the denominator would
> make this task incomparable with the paper. `longvideo_reason_wellformed_accuracy` reports
> the clean 995 beside the headline so the contamination stays visible.

## Frame budget

The videos are long — many run past an hour — so the frame budget matters more here than on a
short-clip benchmark, and it is not fixed by the task. The reference implementation passes the
whole video to the model and leaves sampling to the model wrapper; the paper evaluates
LongVILA-R1 at 512 frames. Pass the budget through the model arguments and **always report it
next to the score**, for example `--model_args ...,max_frames_num=64`.

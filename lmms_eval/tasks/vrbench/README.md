# VRBench

[VRBench](https://vrbench.github.io/) is a multi-step reasoning benchmark over long
narrative videos (ICCV 2025, [arXiv:2506.10857](https://arxiv.org/abs/2506.10857)).
It holds 960 videos of 1.6 hours on average and 8,243 human-labelled questions with
25,106 timestamped reasoning steps. Every question carries exactly four options and
one of seven reasoning types.

The official protocol scores one model response at two levels, so this port ships two
tasks that share `_default_template_yaml` and the same prompt:

* `vrbench_mcq` — outcome level. The option letter is recovered by the regex cascade
  ported from `evaluation/calculate_scores.py:extract_mcq_answer`, and scored as exact
  accuracy. No judge is involved.
* `vrbench_process` — process level. An LLM judge rates the reasoning chain from 0 to 10
  behind a `<rate>` tag, using the prompts in `evaluation/model_api/prompt.py`. The
  routing of `evaluation/run_process_eval.py` is reproduced: `Event Attribution`,
  `Multi-element Inference`, `Implicit Inference` and `Logical Linkage` use the
  unique-answer prompt; `Hypothetical Reasoning` and `Event Prediction` use the
  non-unique-answer prompt and also receive the video summary. `Event Summarization`
  and any other type stay unjudged and are excluded from the process aggregate.
* `vrbench` — the group that runs both tasks and reports the paper's headline number.

Both tasks report a `vrbench_score` on a 0-100 scale plus a per-reasoning-type
breakdown. The group takes the unweighted mean of the two `vrbench_score` values, which
is exactly the paper's `Overall = (MCQ_accuracy + process_score * 10) / 2`.

## Data

The annotations load straight from the Hub, so no manual step is needed:

```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: hf://datasets/OpenGVLab/VRBench/VRBench_eval.jsonl
```

`process_docs` flattens the 960 video-level records into 8,243 question-level documents.

The videos are **not** downloaded automatically. They ship as a 421 GB split zip
(`v001_360p.zip` plus `.z01`-`.z39`) that the generic archive handling cannot unpack.
Download and extract them yourself, then point one env var at the result:

| Variable | Required | Meaning |
|---|---|---|
| `VRBENCH_VIDEO_DIR` | yes | Directory holding the extracted `.mp4` files |
| `VRBENCH_ROOT` | no | Alternative root, searched after `VRBENCH_VIDEO_DIR` |
| `LMMS_EVAL_MEDIA_ROOT` | no | Global media root, searched after the two above |

The annotation field is `VRBench/videos/v001/<video_id>.mp4`, and the resolver also
accepts `<root>/videos/<video_id>.mp4` and `<root>/<video_id>.mp4`.

```bash
hf download OpenGVLab/VRBench --repo-type dataset --local-dir /data/VRBench
cd /data/VRBench/v001_360p_zips && zip -s 0 v001_360p.zip --out v001_360p_joined.zip && unzip v001_360p_joined.zip
export VRBENCH_VIDEO_DIR=/data/VRBench
```

## Judge configuration (process track only)

The judge goes through the shared `lmms_eval.verifiers` layer, so it uses the standard
judge env vars. The paper uses DeepSeek:

```bash
export API_TYPE=openai
export OPENAI_API_URL=https://api.deepseek.com/v1
export OPENAI_API_KEY=<your deepseek key>
export MODEL_VERSION=deepseek-chat   # default when unset
```

A judge call that fails after its retries is dropped from the aggregate instead of
counting as a zero.

## Run

```bash
accelerate launch -m lmms_eval --model <model> --tasks vrbench --batch_size 1
```

Use `--tasks vrbench_mcq` for the rule-based track alone, which needs no API key.
Note that the group runs generation twice, once per task; the official pipeline
generates once and scores the same file twice.

`lmms_eval_specific_kwargs.include_video_summary` defaults to `true`, which matches the
official prompt: the model sees the video and the human-written narrative summary. Set
it to `false` for a video-only ablation.

## Official numbers

From the project page, on the `Overall / MCQ-O / OE-P` scale this port reproduces
(all 0-100, `OE-P` being the 0-10 process score times ten):

| Model | Overall | MCQ-O | OE-P |
|---|---|---|---|
| Gemini-2.0-Pro | 74.61 | 83.29 | 65.93 |
| GPT-4o | 68.68 | 81.23 | 56.13 |
| InternVL2.5-78B | 62.31 | 76.61 | 48.01 |
| Kimi-VL-A3B-Thinking-2506 | 61.82 | 61.67 | 61.97 |
| Qwen2.5-VL-72B | 61.71 | 66.85 | 56.57 |
| Keye-VL-8B-Preview | 60.44 | 64.41 | 56.47 |
| Qwen2.5-VL-7B | 56.52 | 69.61 | 43.43 |

Every question has four options, so random choice scores 25.0 on `vrbench_mcq`.

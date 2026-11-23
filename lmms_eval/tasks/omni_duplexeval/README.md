# Omni-DuplexEval

Omni-DuplexEval ([paper](https://arxiv.org/abs/2605.17360),
[code](https://github.com/OpenBMB/Omni-DuplexEval),
[data](https://huggingface.co/datasets/Hothan/Omni-DuplexEval)) evaluates
**real-time duplex omni-modal interaction**: the model watches a live video
stream while the user speaks an instruction during it, and must either
describe events in real time (with correct timing) or proactively speak at the
right moment (event reminders, post-event reminders, corrections).

| Group | Tasks | Metrics |
| --- | --- | --- |
| `omni_duplexeval_rtd` | 6 configs (world_knowledge, counting, fine_grained_movement, interaction_relation, OCR, Omni) | temporal sensitivity 0-3, content accuracy 0-3 |
| `omni_duplexeval_pr` | `PR_event_reminder`, `PR_post_event_reminder`, `PR_correction` | reminder/correction success 0/1 |

The group `omni_duplexeval` runs everything.

## How it runs

The dataset card states that `question_audio` "is aligned with the full video
duration" and that in PR tasks "the question may occur at arbitrary
timestamps": the track is silent except where the user speaks.
`doc_to_visual` therefore adds the question audio to the video's own sound
track at the same timestamps (ffmpeg, 16 kHz mono sum, frames copied
unchanged) and returns that one video. A duplex wrapper (e.g.
`minicpm_o_45_duplex`) streams it second by second, so the model hears the
user while it watches the video, and returns per-second
`{"text", "current_time", ...}` chunk logs on the video timeline. The
official repository does not include inference code; this input layout
follows the dataset card. `utils.py` ports the official scoring scripts:

- **PR**: for each gold reminder time, the model text inside a 10 s window is
  judged 0/1 (correction tasks judge the full response).
- **RTD**: every sentence gets a temporal-sensitivity score (0-3, multimodal
  judge over ffmpeg-extracted frames) and the whole response a content score
  (0.00-3.00, multimodal judge over the video).

## Judge configuration (identical env vars to the official scripts)

```bash
export DUPLEXEVAL_API_KEY=...     # or OPENAI_API_KEY
export DUPLEXEVAL_BASE_URL=...    # or OPENAI_BASE_URL
export DUPLEXEVAL_MODEL=...      # evaluator model id
```

The RTD temporal judge reads ffmpeg-extracted frames, as in the official
script. The content-accuracy judge sends the official
`{"type": "video", "video": <local path>}` message part, which only endpoints
that can read local files accept. `OMNI_DUPLEXEVAL_CONTENT_INPUT` selects the
input:

| Value | Content judge input |
| --- | --- |
| `auto` (default) | the official video part; if the endpoint rejects it (HTTP 400/415/422), 8 uniformly-sampled frames for the whole run (logged once) |
| `video` | always the official video part |
| `frames` | always 8 uniformly-sampled frames |

Use `frames` for an endpoint that accepts the request but ignores video parts.
Timeouts, rate limits (429) and server errors (5xx) are retried
(`OMNI_DUPLEXEVAL_JUDGE_MAX_ATTEMPTS`, default 3) and never change the input
mode; a call that still fails scores 0, as in the official scripts.
`ffmpeg`/`ffprobe` must be on `PATH`.

## Example

```bash
python -m lmms_eval \
  --model minicpm_o_45_duplex \
  --model_args pretrained=openbmb/MiniCPM-o-4_5 \
  --tasks omni_duplexeval \
  --batch_size 1
```

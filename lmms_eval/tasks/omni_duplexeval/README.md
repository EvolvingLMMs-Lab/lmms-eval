# Omni-DuplexEval

Omni-DuplexEval ([paper](https://arxiv.org/abs/2605.17360),
[code](https://github.com/OpenBMB/Omni-DuplexEval),
[data](https://huggingface.co/datasets/Hothan/Omni-DuplexEval)) evaluates
**real-time duplex omni-modal interaction**: after hearing a spoken
instruction, the model watches a live video stream and must either describe
events in real time (with correct timing) or proactively speak at the right
moment (event reminders, post-event reminders, corrections).

| Group | Tasks | Metrics |
| --- | --- | --- |
| `omni_duplexeval_rtd` | 6 configs (world_knowledge, counting, fine_grained_movement, interaction_relation, OCR, Omni) | temporal sensitivity 0-3, content accuracy 0-3 |
| `omni_duplexeval_pr` | `PR_event_reminder`, `PR_post_event_reminder`, `PR_correction` | reminder/correction success 0/1 |

The group `omni_duplexeval` runs everything.

## How it runs

`doc_to_visual` returns the spoken instruction first and the video second; a
duplex wrapper (e.g. `minicpm_o_45_duplex`) prefills the instruction audio as
the user's speech, then streams the video chunk-by-chunk and returns
per-second `{"text", "current_time", ...}` chunk logs. `utils.py` ports the
official scoring scripts:

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

The RTD content-accuracy judge must accept `{"type": "video"}` message parts;
the remaining judges are text-only. `ffmpeg`/`ffprobe` must be on `PATH`.

## Example

```bash
python -m lmms_eval \
  --model minicpm_o_45_duplex \
  --model_args pretrained=openbmb/MiniCPM-o-4_5 \
  --tasks omni_duplexeval \
  --batch_size 1
```

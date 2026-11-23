# Full-Duplex-Bench (FDB v1.5)

Full-Duplex-Bench evaluates **overlap handling in full-duplex spoken dialogue
models** ([v1.0 paper](https://arxiv.org/abs/2503.04721),
[v1.5 paper](https://arxiv.org/abs/2507.23159),
[code](https://github.com/DanielLin94144/Full-Duplex-Bench),
[data mirror](https://huggingface.co/datasets/Ssshangfu/Full-Duplex-Bench-Data)). A
model must keep listening while it speaks and react correctly when the user
interrupts, backchannels, talks to someone else, or when background speech
appears.

| Task | Samples | Expected behaviour (paper Sec. 2) |
| --- | ---: | --- |
| `fdb_user_interruption` | 200 | answer the new question (`C_RESPOND`) |
| `fdb_user_backchannel` | 98 | keep answering without stopping (`C_RESUME`) |
| `fdb_talking_to_other` | 100 | keep answering (`C_RESUME`); a brief addressee check (`C_UNCERTAIN_HANDLING`) is acceptable |
| `fdb_background_speech` | 100 | keep answering (`C_RESUME`); a brief addressee check (`C_UNCERTAIN_HANDLING`) is acceptable |

The group `full_duplex_bench` runs all four tasks.

## How it runs

Each sample ships a *clean* conversation audio (`clean_input.wav`) and an
*overlapped* variant (`input.wav`). A duplex model wrapper (e.g.
`minicpm_o_45_duplex`) streams both variants as independent sessions and
returns per-second `{"text", "current_time", "is_listen", ...}` chunk logs.
`utils.fdb_process_results` then follows the official v1.5 behaviour
evaluation (`v1_v1.5/evaluation/eval_behavior.py`):

1. model chunks are converted to `{"text", "chunks"}` word-level transcripts,
2. `{input_clean, input_noisy, output_clean, output_noisy}` is sent to an
   OpenAI-compatible judge with the official `behavior.txt` instruction
   (system message) and a `seed`, retrying with `seed + 1` on API errors or
   unparsable replies, as the official script does,
3. the returned `C_*` tag is counted.

## Metrics

| Metric | Meaning |
| --- | --- |
| `fdb_c_respond`, `fdb_c_resume`, `fdb_c_uncertain_handling`, `fdb_c_unknown` | share of each judge tag (the official report) |
| `fdb_expected_behavior` | share of samples whose tag is an expected behaviour for the task (table above); a summary, not an official metric |
| `fdb_scoring_failed` | share of samples that could not be scored (generation error, or the judge still failed after `FDB_JUDGE_MAX_ATTEMPTS` tries, default 5); these samples are left out of the other metrics |

## Deviation from the official pipeline

The official evaluation transcribes the model's *output audio* with an ASR
model that gives word timestamps (parakeet-tdt-0.6b-v2). This task uses the
text that the duplex model emits per one-second streaming chunk and spreads
the words evenly over that second. The judge decides by the position of model
speech relative to the overlap, so the approximate timing can change some
labels. Compare scores only with runs of this task, not with the paper's
tables. The official pipeline also reports stop and response latency, which
this task does not compute.

## Data

The official v1.5 release (MIT License) is a Google Drive folder linked from
the [dataset README](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/main/v1_v1.5/dataset/README.md).
For automatic download the task uses the community mirror
`Ssshangfu/Full-Duplex-Bench-Data`, which holds the same v1.5 archives, and
fetches only its `v1.5/*` files into `$HF_HOME/full_duplex_bench`. The
extraction refuses archive entries that would write outside that folder. To
use a local copy of the official release instead, point `FDB_DATA_ROOT` (or
the task's `dataset_kwargs.data_root`) at a directory that contains
`v1.5/<task>/<sample>/{input,clean_input,...}.{wav,json}`.

## Judge configuration

```bash
export FDB_JUDGE_API_KEY=...        # or DUPLEXEVAL_API_KEY / OPENAI_API_KEY (required)
export FDB_JUDGE_BASE_URL=...       # or DUPLEXEVAL_BASE_URL / OPENAI_BASE_URL (default: OpenAI)
export FDB_JUDGE_MODEL=...          # or DUPLEXEVAL_MODEL / OPENAI_MODEL_ID (default: gpt-4o-2024-08-06, the official judge)
```

The judge only needs a text chat-completions endpoint. Without an API key the
evaluation stops at the first sample instead of scoring every sample as 0.

## Example

```bash
python -m lmms_eval \
  --model minicpm_o_45_duplex \
  --model_args pretrained=openbmb/MiniCPM-o-4_5 \
  --tasks full_duplex_bench \
  --batch_size 1
```

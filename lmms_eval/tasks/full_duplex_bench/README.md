# Full-Duplex-Bench (FDB v1.5)

Full-Duplex-Bench evaluates **overlap handling in full-duplex spoken dialogue
models** ([v1.0 paper](https://arxiv.org/abs/2503.04721),
[v1.5 paper](https://arxiv.org/abs/2507.23159),
[code](https://github.com/DanielLin94144/Full-Duplex-Bench),
[data](https://huggingface.co/datasets/Ssshangfu/Full-Duplex-Bench-Data)). A
model must keep listening while it speaks and react correctly when the user
interrupts, backchannels, talks to someone else, or when background speech
appears.

| Task | Samples | Correct behaviour |
| --- | ---: | --- |
| `fdb_user_interruption` | 200 | answer the new question (`C_RESPOND`) |
| `fdb_user_backchannel` | 98 | keep answering without stopping (`C_RESUME`) |
| `fdb_talking_to_other` | 100 | keep answering (`C_RESUME`) |
| `fdb_background_speech` | 100 | keep answering (`C_RESUME`) |

The group `full_duplex_bench` runs all four tasks.

## How it runs

Each sample ships a *clean* conversation audio (`clean_input.wav`) and an
*overlapped* variant (`input.wav`). A duplex model wrapper (e.g.
`minicpm_o_45_duplex`) streams both variants as independent sessions and
returns per-second `{"text", "current_time", "is_listen", ...}` chunk logs.
`utils.fdb_process_results` then follows the official v1.5 evaluation:

1. model chunks are converted to word-level timestamped transcripts,
2. `{input_clean, input_noisy, output_clean, output_noisy}` is sent to an
   OpenAI-compatible judge with the official `behavior.txt` instruction,
3. the returned `C_*` label is compared against the per-task correct label.

`fdb_behavior_acc` is the official accuracy; `fdb_responded_after_overlap` is
a judge-free sanity metric (model produced speech after the overlap began).

## Data

The task downloads and unzips the v1.5 archives from
`Ssshangfu/Full-Duplex-Bench-Data` into `$HF_HOME/full_duplex_bench` on first
use. To reuse a pre-existing extraction, point `FDB_DATA_ROOT` (or the task's
`dataset_kwargs.data_root`) at a directory that contains
`v1.5/<task>/<sample>/{input,clean_input,...}.{wav,json}`.

## Judge configuration

```bash
export FDB_JUDGE_API_KEY=...        # or DUPLEXEVAL_API_KEY / OPENAI_API_KEY
export FDB_JUDGE_BASE_URL=...       # or DUPLEXEVAL_BASE_URL / OPENAI_BASE_URL
export FDB_JUDGE_MODEL=...          # or DUPLEXEVAL_MODEL / OPENAI_MODEL_ID
```

The judge only needs a text chat-completions endpoint.

## Example

```bash
python -m lmms_eval \
  --model minicpm_o_45_duplex \
  --model_args pretrained=openbmb/MiniCPM-o-4_5 \
  --tasks full_duplex_bench \
  --batch_size 1
```

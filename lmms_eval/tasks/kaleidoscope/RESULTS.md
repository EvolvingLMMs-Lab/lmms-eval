# Kaleidoscope — reproduction notes

## 1. Answer extraction vs. the reference implementation

The reference repo publishes raw and post-processed model outputs for two models
under `sample_outputs/zero-shot/` (100 questions each). Running this task's
`extract_choice` over the raw `reasoning` field and comparing against the
reference `format_answer.py` output in `results_format.json`:

| Model | Agreement |
|---|--:|
| Qwen2.5-VL-3B | **100 / 100** |
| Aya-Vision | **99 / 100** |

The single Aya-Vision difference is the reference's line-bound `{"choice": ...}`
regex failing on

```json
{
  "choice": "C",
  "reasoning": "The traffic interchange depicted in the image is ..."
}
```

which the reference counts as a format error and this task counts as a valid
answer. See fidelity note 4 in `README.md`.

## 2. End-to-end reproduction on the reference's 100-question slice

The reference's published sample is the **first 100 rows** of
`CohereLabs/kaleidoscope` (all English, all multimodal, GATE 2022). Those rows
are order-identical to `--tasks kaleidoscope_multimodal --limit 100`, so the two
runs score exactly the same questions.

| Run | Decoding | acc | valid_acc | format_error |
|---|---|--:|--:|--:|
| **Reference** (`sample_outputs/model_qwen2.5-3b`, vLLM) | T=0.7, top_p=0.9 | **30.00** | 30.00 | 0.00 |
| lmms-eval, this task (HF transformers) | greedy | **31.00** | 31.00 | 0.00 |
| lmms-eval, this task | T=0.7, seed 1234 | 37.00 | 38.14 | 3.00 |
| lmms-eval, this task | T=0.7, seed 7 | 32.00 | 33.33 | 4.00 |
| lmms-eval, this task | T=0.7, seed 99 | 35.00 | 35.71 | 2.00 |

Greedy decoding lands within **one question** of the reference (31 vs 30). The
sampled runs scatter over 32–37, which is what `temperature=0.7` on `n=100`
buys you (binomial s.e. ≈ 4.6 pp) — the paper's sampling parameters are kept as
the task default for fidelity, but `--gen_kwargs temperature=0,do_sample=False`
is the reproducible setting.

Commands:

```bash
python -m lmms_eval --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks kaleidoscope_multimodal --batch_size 1 --limit 100 \
    --gen_kwargs temperature=0,do_sample=False
```

## 3. Paper comparison target (full multimodal split)

The paper's appendix table (`tables/complete_lang_acc.tex`, multimodal split)
reports the following for **Qwen2.5-VL-3B** — the numbers a full
`kaleidoscope_multimodal` run should be compared against. This task's
`kaleidoscope_acc` aggregation prints the same breakdown, so one run reproduces
the whole table.

| | en | fr | de | nl | pt | es | ar | bn | hr | hi | hu | lt | ne | fa | ru | sr | te | uk |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Total Acc. | 35.0 | 32.3 | 21.1 | 41.8 | 52.2 | 53.3 | 33.5 | 32.8 | 25.9 | 31.3 | 28.2 | 35.6 | 23.8 | 30.3 | 29.0 | 27.8 | 31.8 | 40.1 |
| Valid Acc. | 35.0 | 32.4 | 21.1 | 41.8 | 52.3 | 53.3 | 33.7 | 32.8 | 26.4 | 31.3 | 29.2 | 35.6 | 23.8 | 30.4 | 29.0 | 28.1 | 31.8 | 40.1 |
| FE | 0.0 | 0.3 | 0.0 | 0.0 | 0.1 | 0.0 | 0.5 | 0.2 | 1.9 | 0.0 | 3.2 | 0.0 | 0.0 | 0.2 | 0.0 | 1.1 | 0.0 | 0.0 |

Macro-average over the 18 languages: **Total Acc. 33.7**, **Valid Acc. 33.8**.

```bash
python -m lmms_eval --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks kaleidoscope_multimodal --batch_size 1 \
    --log_samples --output_path ./logs/
```

## 4. Environment used

| | |
|---|---|
| GPU | 1× RTX 3060 (12 GB) |
| Python | 3.10 (conda) |
| torch | 2.13.0+cu126 |
| transformers | 5.14.1 |
| datasets | 5.0.0 |
| lmms-eval | 0.7.2 (`v0.6-167-g6619ef61`) |

Unit tests: `pytest test/eval/test_kaleidoscope.py` — 44 passed.

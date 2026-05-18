# feat: add XModBench cross-modal benchmark + interleaved-multimedia omni model wrappers

## Summary

Adds **XModBench** — a tri-modal (audio / vision / text) cross-modal
consistency MCQ benchmark (5 task families, 17 subtasks, 61,320 QA pairs) —
plus a balanced **XModBench-Lite** split (6,000 samples: 5 families × 6
modality configurations × 200) and the chat-style model wrappers needed to
evaluate omni models on it.

Data is hosted on the Hugging Face Hub (`RyanWW/XModBench`) and auto-resolved
via `snapshot_download`; no data is vendored in the repo.

## Why the new model wrappers

XModBench items carry media in **both** the question stem **and every answer
option** (up to 5 media per item). lmms-eval's *simple* model interface
attaches only one media object per request via `doc_to_visual`, so an
omni model silently sees just the first media and scores near chance.

This PR adds *chat-style* (`is_simple = False`) wrappers that consume the
task's `doc_to_messages` output and feed the full interleaved prompt to the
model:

- `lmms_eval/models/chat/_interleave_base.py` — shared request loop
  (`InterleaveChatMixin`); per-model `video_kwargs`/`image_kwargs` budget.
- `qwen2_5_omni_interleave`, `qwen3_omni_interleave` — `process_mm_info` path.
- `omnivinci_interleave` — VILA processor path.
- `baichuan_omni_interleave` — special-token string-prompt path.

**No upstream model file is modified.** The only change outside new files is
4 registry lines in `lmms_eval/models/__init__.py`.

## Validation (XModBench-Lite vs. paper, ICLR 2026 Table 2)

| Config | Qwen2.5-Omni | paper | Δ | Qwen3-Omni | Baichuan-1.5 | paper | Δ | OmniVinci |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A→T | 63.1 | 62.0 | +1.1 | 71.6 | 52.5 | 47.8 | +4.7 | 62.2 |
| A→V | 49.8 | 48.0 | +1.8 | 52.0 | 32.0 | 35.8 | −3.8 | — |
| T→A | 59.2 | 55.4 | +3.8 | 62.5 | 47.6 | 40.5 | +7.1 | — |
| T→V | 62.5 | 59.6 | +2.9 | 67.0 | 56.6 | 56.2 | +0.4 | — |
| V→A | 50.3 | 50.5 | −0.2 | 55.6 | 47.0 | 38.6 | +8.4 | — |
| V→T | 76.4 | 76.3 | +0.1 | 83.1 | 77.7 | 73.0 | +4.7 | 78.8 |

- **Qwen2.5-Omni: 6/6 configs within |Δ|<5** — paper reproduced on Lite.
- **Qwen3-Omni**: not in the paper (newer model); first numbers reported.
- **Baichuan-Omni-1.5: 4/6 within |Δ|<5**; T→A/V→A are genuine positive
  Lite-subsample deviations (clean runs).
- **OmniVinci: 2/6 best-effort**; the other 4 hit VILA-internal limits under
  interleaved multi-media prompts (documented in `RESULTS.md`).
- **VITA**: skipped — its wrapper only supports a single media tensor per
  request (architectural).

## What's included

- `lmms_eval/tasks/xmod_bench/` — task configs (full + Lite), `utils.py`
  (HF auto-download, doc_to_messages, Level-1 aggregation),
  `make_lite.py` (Lite generator), `summarize.py` (Level-2: by-config,
  by-family, modality disparity, directional imbalance),
  `README.md`, `RESULTS.md`.
- `lmms_eval/models/chat/*_interleave.py` + base mixin.
- Launchers: `submit_lite.sh`, `run_xmod_lite_generic.slurm`, etc.

## Reproduce

```bash
./submit_lite.sh qwen2_5_omni_interleave Qwen/Qwen2.5-Omni-7B qwenomni3
python lmms_eval/tasks/xmod_bench/summarize.py \
  --logs logs/xmod_bench_lite/results_qwen2_5_omni_interleave/
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# vending_bench2_seed

This is a no-sandbox seed integration for Vending-Bench 2 style agentic loop evaluation in lmms-eval.

- Output type: `generate_until_agentic`
- Tools: pure Python simulator functions in `utils.py`
- Loop: model emits `<tool_call>...</tool_call>` and eventually `<submit>...</submit>`
- Goal: satisfy target constraints (`min_cash`, `min_days_elapsed`) before submit

Quick smoke run:

```bash
python -m lmms_eval \
  --model openai \
  --model_args model=gpt-4o-mini \
  --tasks vending_bench2_seed \
  --limit 2 \
  --batch_size 1
```

Notes:

- This seed task validates `generate_until_agentic` with deterministic vending simulation.
- Full official Vending-Bench 2 integration can be layered on top by replacing the seed JSONL with official task contracts.

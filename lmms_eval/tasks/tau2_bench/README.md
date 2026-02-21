# tau2_bench_telecom_seed

This is a no-sandbox seed integration for agentic loop evaluation in lmms-eval.

- Output type: `generate_until_agentic`
- Tools: pure Python functions in `utils.py`
- Loop: model emits `<tool_call>...</tool_call>` and eventually `<submit>...</submit>`
- Goal: reach `target_state` before submit

Quick smoke run:

```bash
python -m lmms_eval \
  --model openai \
  --model_args model=gpt-4o-mini \
  --tasks tau2_bench_telecom_seed \
  --limit 2 \
  --batch_size 1
```

Notes:

- This seed task validates the new agentic infrastructure (multi-step tool calls + submit).
- Full official tau2-bench domain integration can be layered on top by converting official task specs to lmms-eval docs.

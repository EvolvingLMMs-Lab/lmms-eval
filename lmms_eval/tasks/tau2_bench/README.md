# tau2_bench_telecom

τ2-Bench telecom agentic loop evaluation in lmms-eval.

- Output type: `generate_until_agentic`
- Tools: pure Python functions in `utils.py`
- Loop: model emits `<tool_call>...</tool_call>` and eventually `<submit>...</submit>`
- Goal: reach `target_state` before submit

Quick smoke run:

```bash
python -m lmms_eval \
  --model openai \
  --model_args model=gpt-4o-mini \
  --tasks tau2_bench_telecom \
  --limit 2 \
  --batch_size 1
```

Notes:

- Validates the agentic infrastructure (multi-step tool calls + submit).
- Full official τ2-bench domain integration can be layered on top by converting official task specs to lmms-eval docs.

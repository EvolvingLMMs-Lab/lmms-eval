# Evaluate Your Model in 5 Minutes

This quick-start gives you a fast path from clone to first successful run.

## 1) Clone and install

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
uv sync
```

## 2) Run a smoke evaluation

```bash
uv run python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

## 3) Read the output

- If the command completes and prints metrics, your environment is ready.
- If it fails, capture the full traceback and open a bug report using the issue
  form.

## 4) Next steps

- Add your own task list: `--tasks mmmu,mme`
- Switch model weights via `--model_args`
- Explore guides:
  - `docs/task_guide.md`
  - `docs/model_guide.md`
  - `CONTRIBUTING.md`

## Need help?

- Discord: <https://discord.gg/zdkwKUqrPy>
- Issues: <https://github.com/EvolvingLMMs-Lab/lmms-eval/issues>

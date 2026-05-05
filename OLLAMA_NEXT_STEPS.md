# Ollama Backend — Next Steps

## Where we left off

Branch: `feat/ollama-model`

Two files were added:

- `lmms_eval/models/chat/ollama.py` — the backend
- `test/models/test_ollama.py` — unit tests (10/10 passing, no Ollama required)

One line was added to `lmms_eval/models/__init__.py`:
```python
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    "ollama": "Ollama",   # ← added
    ...
}
```

---

## What still needs to be done

### 1. Install Ollama and pull a vision model

```bash
# Install from https://ollama.com
ollama serve                  # start the server (runs on localhost:11434)
ollama pull llava             # smallest vision model, good for smoke testing
ollama pull llava-llama3      # better quality if you have the VRAM
```

### 2. Run a live smoke test

```bash
uv run python -m lmms_eval \
    --model ollama \
    --model_args model_version=llava \
    --tasks mme \
    --limit 8
```

Expected: 8 samples evaluated, scores printed. If it errors, check:
- Is `ollama serve` running?
- Does `ollama list` show `llava`?
- Is the `logprobs` field actually present in `/api/generate` responses for your Ollama version?

### 3. Verify loglikelihood against the real API

The `loglikelihood` implementation uses `POST /api/generate` with `logprobs=True`.
This field was added in Ollama v0.1.38. Confirm it works:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llava",
  "prompt": "The sky is blue",
  "stream": false,
  "logprobs": true
}'
```

The response should contain a `"logprobs"` array of floats. If it's missing or null,
the implementation will silently return `-inf` for all loglikelihood requests — tasks
that depend on it (e.g. multiple-choice scoring) will produce wrong results.

### 4. Run pre-commit linting (required before PR)

```bash
uv run pip install pre-commit
uv run pre-commit install
uv run pre-commit run --all-files
```

This runs Black (line length 240) + isort. Fix any formatting issues it flags.

### 5. Commit

```bash
git add lmms_eval/models/chat/ollama.py \
        lmms_eval/models/__init__.py \
        test/models/test_ollama.py
git commit -m "feat: add Ollama local inference backend"
```

### 6. Open a pull request

The upstream repo is `EvolvingLMMs-Lab/lmms-eval`. You'll need to fork it if you
haven't already, push the branch, and open a PR against `main`.

PR description should include:
- What model this adds and why (local inference, no API key, multimodal)
- Supported Ollama models (llava, llava-llama3, moondream, minicpm-v, ...)
- Known limitations (no video/audio, logprobs requires Ollama ≥ v0.1.38)
- Example usage command

---

## Known limitations to mention in the PR

| Feature | Status |
|---|---|
| Image input | Works (base64 via OpenAI-compat /v1) |
| Text-only models | Works |
| Video input | Not supported by Ollama |
| Audio input | Not supported by Ollama |
| `loglikelihood` | Works if Ollama ≥ v0.1.38 and model supports it |
| `generate_until_multi_round` | Inherited from OpenAI parent (raises NotImplementedError) |

---

## Quick reference

```bash
# Set up on new machine
git clone <your-fork>
cd lmms-eval
git checkout feat/ollama-model
uv sync

# Run unit tests (no Ollama needed)
uv run python -m pytest test/models/test_ollama.py -v

# Run live eval (Ollama must be running)
uv run python -m lmms_eval \
    --model ollama \
    --model_args model_version=llava \
    --tasks mme --limit 8
```

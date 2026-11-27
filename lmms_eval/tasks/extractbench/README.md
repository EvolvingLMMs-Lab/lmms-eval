# ExtractBench

Schema-guided extraction from the original PDFs in
[LlamaIndex ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench).
This is the ordinary 370-document benchmark, not ExtractBench Pro or the
unrelated Contextual AI benchmark with the same name.

| Task | Documents | Pages per document |
| --- | ---: | --- |
| `extractbench_short` | 252 | 1–10 |
| `extractbench_medium` | 98 | 11–50 |
| `extractbench_long` | 20 | More than 50 |
| `extractbench` | 370 | All three splits |

Dataset revision: `f9b910906cd7e73cc9eb5379a8d5d935c67b92c4`.
Evaluator and prompt: the published
[llama-extract-bench 1.0.0](https://pypi.org/project/llama-extract-bench/1.0.0/)
release (Apache-2.0), with `parse-bench==2.0.0` for its shared schema models.

## Protocol

This task defines a **full-document, single-turn adaptation**. Every PDF page is
rendered by Poppler at 150 DPI and sent in source order. No page selection, OCR
text, gold field rules, expected values or evaluation-only row identities enter
the prompt. The schema is transformed with the official repeated-structure
promotion and `additionalProperties: false` helpers.

Chat backends receive the official system prompt and a user message containing
all images followed by the official schema prompt. Simple backends concatenate
the system and user text, so that interface is a separate role-flattening
adaptation. The default output budget is 32,768 tokens, temperature 0, with no
text stop sequences. Record backend image resizing, context limits and thinking
mode with results; the task's full images do not override model preprocessing.

The official leading `<think>` removal, JSON parser, truncated-JSON salvage and
depth pruning are reused. Salvage can recover partial credit from incomplete or
fenced JSON. `parse_success` measures syntactic parsing, not schema validity or
correctness; `salvaged` reports recovery after initial JSON parsing failed.
Framework reasoning-tag removal is disabled.

The official Qwen pipeline also uses JSON-object constrained decoding, adaptive
context fitting and a benchmark runner with per-document deadlines. This task
does **not** reproduce those controls. A transport timeout is not an end-to-end
document deadline. It never silently drops pages or schema fields to fit a
context window. Results are comparable within the declared adaptation, not
automatically to official leaderboard scores. In particular, long documents may
exceed a backend's context or image limits.

## Scores

The pinned `ExtractEvaluator.compute_metrics` supplies
`extract_unified_value_f1`, including its list and EOB normalization, nested
record alignment and alternative accepted values. The task exposes:

- `extractbench_value_f1`: arithmetic mean of document Value F1, in [0, 1].
- `extractbench_perfect`: fraction of documents whose Value F1 is exactly 1.
- `extractbench_parse_success`: fraction of responses parsed by the official parser.
- `extractbench_salvaged`: fraction recovered after an initial JSON parse failure.

Every evaluated document contributes to all four denominators. Empty,
unparseable or unscoreable model output has Value F1 0. An official evaluator
result without a Value F1 also contributes 0; it is not omitted. Configuration,
data, dependency and scorer failures raise errors rather than become model
scores. The task also rejects the OpenAI backend's `LMMS_EVAL_REQUEST_FAILED`
and `LMMS_EVAL_BUDGET_EXCEEDED` sentinels. Request failures can produce these
markers even in strict mode.
Other backends may replace infrastructure failures with empty strings; those
cannot be distinguished from genuine empty answers at this scoring boundary.
Inspect captured requests/responses for context errors or truncation before
accepting results.

The group weights split means by document count: the full denominator is 370,
not three equally weighted split scores. This is document macro F1, not a
pooled cell-level F1. The base prompt does not request citations; grounding
scores are deliberately omitted.

## Setup and execution

Python 3.12 or newer is required by the official scorer. From the source checkout:

```bash
uv sync --python 3.12 --extra extractbench
```

The runtime guard checks the scorer and shared schema package versions.
The dependency lock records the release artifact hashes.

Install Poppler separately (`brew install poppler` on macOS, or the
`poppler-utils` system package on Debian/Ubuntu). The task does not install
software during evaluation. Dataset JSONL files and PDFs use the normal
Hugging Face cache; prefetch them before an offline run.

For an OpenAI-compatible vision endpoint, select the **chat** OpenAI backend
with strict generation. On this backend, `max_retries=1` means one attempt:

```bash
LMMS_IMAGE_ENCODE_FORMAT=PNG uv run --extra extractbench python -m lmms_eval \
    --model openai \
    --model_args model=YOUR_MODEL,base_url=http://localhost:8000/v1,strict_generation=true,timeout=1800,max_retries=1 \
    --tasks extractbench --batch_size 1 --log_samples --output_path ./logs/extractbench
```

Check the selected backend's CLI and model requirements. Select a leaf task for
a split-specific run. `--limit` is useful for debugging but does not establish
full benchmark coverage. Keep raw responses and record the dataset revision,
scorer version, resolved dependencies, model revision, mode and effective
generation settings with every result.

## Tests

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    uv run --extra extractbench pytest test/tasks/test_extractbench.py -q
```

Tests use synthetic rows, images and PDFs, with no model or dataset download.
Scorer integration tests skip if the optional package is absent; the command
above installs it and exercises them. PDF integration requires Poppler.

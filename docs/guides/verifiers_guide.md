# Verifiers Guide

`lmms_eval.verifiers` is a small, pluggable module for deciding whether a model's prediction is correct. It sits between raw model output and `process_results`, and it complements (does not replace) the batch-level `Filter` system in `lmms_eval.filters` - filters operate on a full response list, verifiers operate per-sample.

## Two-Layer Abstraction

| Layer | Role | Location |
|-------|------|----------|
| **Extractor** | Per-sample text transform applied before verification (strip reasoning, regex, MCQ letter, number) | `lmms_eval/verifiers/extractors.py` |
| **Verifier** | Judges correctness given `(question, prediction, ground_truth)` and returns a `VerifyResult` | `lmms_eval/verifiers/rule.py`, `openai.py`, `gemini.py`, `composite.py` |

Both are composed by a `VerificationPipeline`:

```
raw model output -> extractor 1 -> extractor 2 -> verifier -> VerifyResult
```

## Quick Start

```python
from lmms_eval.verifiers import VerificationPipeline
from lmms_eval.verifiers.extractors import StripReasoningExtractor
from lmms_eval.verifiers.rule import ExactMatchVerifier

pipeline = VerificationPipeline(
    extractors=[StripReasoningExtractor()],
    verifier=ExactMatchVerifier(ignore_case=True),
)
result = pipeline("What color is the sky?", "<think>hmm</think>Blue", "Blue")
assert result.is_correct
```

Or build one from a config dict (handy for YAML-driven tasks):

```python
from lmms_eval.verifiers import build_verification_pipeline

pipeline = build_verification_pipeline({
    "extractors": [{"type": "strip_reasoning"}],
    "verifier": {"type": "openai", "model": "gpt-4o-2024-11-20", "judge_type": "binary"},
})
```

## Built-in Extractors

| Type | Class | What it does |
|------|-------|---------------|
| `strip_reasoning` | `StripReasoningExtractor` | Removes `<think>...</think>` / `<thinking>...</thinking>` blocks |
| `regex` | `RegexExtractor` | Returns the first regex capture group, with a fallback default |
| `mcq` | `MCQExtractor` | Extracts a multiple-choice letter from common answer formats |
| `number` | `NumberExtractor` | Extracts `\boxed{...}` content or the last number in the text |
| `strip` | `StripExtractor` | Strips whitespace and common special tokens (`</s>`, `<\|im_end\|>`, ...) |

## Built-in Verifiers

| Type | Class | What it does |
|------|-------|---------------|
| `exact_match` | `ExactMatchVerifier` | Case/whitespace-normalized exact string match |
| `contains` | `ContainsVerifier` | Ground truth appears as a substring of the prediction |
| `mcq_match` | `MCQMatchVerifier` | Multiple-choice letter match, tolerant of `(A)` / `A.` / `A)` formatting |
| `numeric` | `NumericToleranceVerifier` | Numeric comparison within relative/absolute tolerance |
| `openai` | `OpenAIVerifier` | LLM-as-judge via `lmms_eval.llm_judge`; `judge_type` is `binary` or `comparative`, with score-style output set by `response_format` and custom prompts by `custom_prompt` |
| `gemini` | `GeminiVerifier` | LLM-as-judge via the Gemini SDK, including multimodal (text + image) judging |
| `composite` | `CompositeVerifier` | Chains verifiers, falling through to the next when one reports low confidence |

`openai` and `gemini` are lazily imported so the module has no hard dependency on either SDK unless you actually instantiate that verifier.

### Composite Fallback

Run a cheap rule-based check first, and only pay for an LLM judge when the cheap check is uncertain. A verifier signals uncertainty via `result.metadata["confident"] = False`; the composite verifier moves to the next one in the chain when it sees that flag, and always accepts the last verifier's result regardless of confidence. A result with no `confident` key counts as confident (the chain stops), so every rule verifier sets the key explicitly.

Rule verifiers follow a two-tier convention for `confident`:

- **String-match** verifiers (`exact_match`, `contains`) treat any non-match as low-confidence, so a mismatch falls through to the next verifier.
- **Parse-based** verifiers (`mcq_match`, `numeric`) treat only a *parse failure* as low-confidence — `mcq_match` when no single MCQ letter can be extracted, `numeric` when either the prediction or ground truth can't be parsed as a number. A cleanly-parsed but wrong answer is a *confident wrong* and does **not** fall through.

```python
from lmms_eval.verifiers.composite import CompositeVerifier
from lmms_eval.verifiers.rule import ExactMatchVerifier
from lmms_eval.verifiers.openai import OpenAIVerifier

verifier = CompositeVerifier([
    ExactMatchVerifier(),   # fast, always confident
    OpenAIVerifier(),       # expensive fallback
])
```

### Judge Failures

The LLM-judge verifiers (`openai`, `gemini`) never raise out of `verify`. When the judge call fails — an exception, retry exhaustion, or the provider reporting `success=False` — they return a `VerifyResult` with `score=0.0` / `is_correct=False` **and** `metadata["judge_failed"] = True`. The score stays 0 so nothing downstream breaks, but the flag lets a task tell an infra failure apart from a genuinely wrong prediction, instead of silently counting it as wrong:

```python
def my_process_results(doc, results):
    verdict = _pipeline(doc["question"], results[0], doc["answer"])
    if verdict.metadata.get("judge_failed"):
        # judge infra failed — track separately so it doesn't count against acc
        return {"acc": 0.0, "judge_failed": 1.0}
    return {"acc": 1.0 if verdict.is_correct else 0.0, "judge_failed": 0.0}
```

## Using it in a Task

Call the pipeline from `process_results` in your task's `utils.py`, same as any other scoring helper:

```python
from lmms_eval.verifiers import build_verification_pipeline

_pipeline = build_verification_pipeline({
    "extractors": [{"type": "strip_reasoning"}],
    "verifier": {"type": "exact_match"},
})

def my_process_results(doc, results):
    prediction = results[0]
    verdict = _pipeline(doc["question"], prediction, doc["answer"])
    return {"acc": 1.0 if verdict.is_correct else 0.0}
```

## Writing a Custom Verifier or Extractor

Subclass `Verifier` or `Extractor` from `lmms_eval.verifiers.base` and implement `verify` / `extract`:

```python
from lmms_eval.verifiers.base import Verifier, VerifyResult

class MyVerifier(Verifier):
    def verify(self, question, prediction, ground_truth, **kwargs) -> VerifyResult:
        correct = my_custom_check(prediction, ground_truth)
        return VerifyResult(score=1.0 if correct else 0.0, is_correct=correct)
```

To make it available to `build_verification_pipeline` via a config `"type"` string, register it in `VERIFIER_REGISTRY` (or `EXTRACTOR_REGISTRY`) from `lmms_eval.verifiers`.

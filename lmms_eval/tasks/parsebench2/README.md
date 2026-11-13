# ParseBench-2

`--tasks parsebench2` evaluates document transcription and structured layout through
lmms-eval's native `generate_until` interface, for simple and chat model adapters.
It loads all 4,317 documents in the pinned public snapshot, interleaving these 13
subsets: text, headings, strikeout, underline, tables, complex_table_headers,
financial_tables, insurance_filing_tables, dense_tables, charts, layout, forms,
and fillable_forms. `--limit 13` selects one document per subset; it is a smoke
sample, not a benchmark result.

Sources: [dataset](https://huggingface.co/datasets/llamaindex/ParseBench-2),
[published outputs](https://huggingface.co/datasets/llamaindex/ParseBench-2-results),
[official scorer](https://github.com/run-llama/ParseBench/tree/v2.0.1).

## Install and run

Python 3.12 or newer is required for this optional task. Other tasks retain the
harness's existing Python support. Scorer/provider packages are imported lazily;
task discovery does not require them.

```bash
uv sync --python 3.12 --extra parsebench2
uv run --extra parsebench2 python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks parsebench2 --batch_size 1 --limit 13 \
  --log_samples --output_path ./logs/parsebench2
```

This is an example command, not a validated fresh-model result. For reproducible
comparisons, pin the model checkpoint revision through the chosen backend and
save the model/backend version, model args, generation settings, and harness
commit alongside the sample logs. Backend resizing and image limits also matter.
The example model name alone is not a frozen model version.

The optional extra includes the official scorer's `llamaparse` extra because its
normalized-layout adapter imports SDK types. Scoring and normalization make no
provider API calls. No LLM judge is enabled. Files are downloaded through the
Hugging Face cache; nothing is installed automatically during evaluation.

## Input and response contract

Only `example_id`, `subset`, and the source `document` path enter the harness's
dataset. Reference Markdown, rules, metadata, and tags are excluded. The fixed
prompt never reads labels. Reference sidecars are loaded only during scoring.
Manifest IDs, rather than cache filenames, identify documents. Each PDF supplies
all pages as RGB images rendered with `pypdfium2==4.30.0` at scale 2 (144 DPI).
Original image inputs are decoded as RGB. No page selection uses annotations.

The integration prompt `lmms_eval_parsebench2_v1` requests a JSON object:

```json
{
  "markdown": "# Heading\n\nText",
  "layout_pages": [{
    "page_number": 1, "width": 1, "height": 1,
    "items": [{
      "type": "heading", "md": "# Heading",
      "bbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.1}
    }]
  }]
}
```

Coordinates are fractional xywh in the unrotated page frame. The native adapter
validates that region boxes fit in the page and maps each region box to one
attributed segment before calling the official normalized-layout adapter. Page
Markdown defaults to the concatenated item Markdown. Native items can also carry
the official `layout_segments` and granular layers; coordinate interpretation
then follows the official IR. These are predictions, never inferred from labels.
This response contract/prompt/rendering is an lmms-eval adaptation, not a claim to
reproduce a published provider's prompt or leaderboard entry.

Plain Markdown scores applicable transcription dimensions. If any evaluated
layout document lacks an explicit `layout_pages` channel, layout and overall are
reported as `NaN` (unavailable). An explicit empty `layout_pages: []` declares a
supported channel with no predictions and receives the official zero layout
score. `parsebench2_layout_coverage` reports the fraction of evaluated layout
documents declaring the channel. Do not describe Markdown-only results as
complete six-dimension PB2 scores.

A full official `InferenceResult` JSON record is also accepted for replay. Request
and normalized-output IDs and pipeline names must agree; wrong-document records
are rejected. `raw_output` and normalized output are preserved because provider
layout adapters may use both. Published records must be identified as replay,
not as newly generated responses from the selected model.

The default filter selects the first generation. No task-specific reasoning
removal or whitespace stripping occurs. JSON fences explicitly labelled `json`
are accepted; arbitrary prose is not mined for embedded JSON. Missing, empty,
malformed JSON and invalid native schemas become empty predictions with an
explicit failure status and remain in the scoring denominator. Scorer/data or
replay identity errors stop evaluation instead of dropping documents. The task
requires a single filtered response per document. `--log_samples` retains the
response, failure status, layout support and official document metrics.

## Scoring and versions

Scoring is pinned to `parse-bench==2.0.1`, dataset revision
`a12c9785cd8cd015b5284e0ad0d74722091762f2`. YAML metadata records those versions,
the prompt version, and renderer version. The task calls official
`compute_next_metrics`, `EvaluationRunner.aggregate_metrics` and
`combine_next_score`; no metric formulas or weighting are reimplemented.

`parsebench2_text`, `parsebench2_formatting`, `parsebench2_tables`,
`parsebench2_charts`, `parsebench2_layout`, and `parsebench2_forms` use the official
subset macro averages and dimension weights. `parsebench2_overall` is the
official unweighted mean of all six complete dimensions. Missing subsets/metrics
leave affected dimensions and overall unavailable. Scores are on [0,1].
`parsebench2_response_failure_rate` and `parsebench2_layout_coverage` are integration
diagnostics, not extra official PB2 dimensions. Timing placeholders used to
construct native scorer records are not measured latency or cost.

## CPU verification without inference

```bash
uv run --extra parsebench2 python -m pytest test/tasks/test_parsebench2.py -q
uv run --extra parsebench2 python scripts/validation/parsebench2_replay.py \
  --per-subset 10 --output /tmp/pb2-replay
```

The public replay helper downloads `azure_di_layout` outputs pinned to revision
`80df59e43b88c4474a1c9b90d0315edc70b2b1f8`, selects the first 10 manifest documents
per subset, renders their media, runs task scoring, and compares the entire
combined report (including every slot and document count) with the official CLI.
Scores allow absolute tolerance `1e-12` for parallel summation order; structure
and counts must match exactly.
It checks official successful/failed/skipped counts and writes a receipt plus a
`from_log` input. It prints a public `lmms_eval --model from_log` command to exercise
request generation, filtering, sample logging and aggregation without inference.
`from_log` does not render media itself; the helper renders separately.

These checks establish only the tested sample's scoring/transport parity and
local harness behavior. They do not establish full-run parity on 4,317 documents,
fresh-model inference success, or leaderboard equivalence. Keep an integration
PR in draft until the repository's real-model E2E requirement is satisfied.

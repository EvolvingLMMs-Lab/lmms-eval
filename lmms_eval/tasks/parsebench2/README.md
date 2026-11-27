# ParseBench-2

`--tasks parsebench2` evaluates document transcription and structured layout through
lmms-eval's native `generate_until` interface, for simple and chat model adapters.
Its manifest contains 4,317 documents in the pinned public snapshot, interleaving
these 13 subsets: text, headings, strikeout, underline, tables, complex_table_headers,
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
  --tasks parsebench2 --batch_size 1 \
  --log_samples --output_path ./logs/parsebench2 --fail_on_error
```

The `parsebench2` and `extractbench` extras pin different `parse-bench` versions
(2.0.1 and 2.0.0) and are declared as conflicting for uv; install them in separate
environments. For reproducible comparisons, pin the model checkpoint revision
through the chosen backend and save the model/backend version, model args,
generation settings, and harness commit alongside the sample logs.

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
then follows the official IR. This response contract/prompt/rendering is an
lmms-eval adaptation, not a claim to reproduce a published provider's prompt or
leaderboard entry.

Each response gets a `response_status`, logged with `--log_samples`:

| Status | Response | Scored prediction |
|---|---|---|
| `structured` | valid JSON object | its Markdown and layout |
| `invalid_layout` | valid JSON, layout outside the contract (box off the page, unnormalized page) | its Markdown, no layout |
| `markdown_only` | does not start with a JSON object | the response as Markdown |
| `invalid` | JSON that does not decode (cut off, unescaped newline, ...) | the `"markdown"` string recovered from it, else empty |
| `missing` | empty | empty |
| `official_record` | a full official `InferenceResult` (replay) | the record as is |

A leading code fence (```` ``` ```` or ```` ```json ````) is removed first, also when
the response stops before the closing fence. Arbitrary prose is not mined for
embedded JSON. `parsebench2_response_failure_rate` counts `missing`, `invalid` and
`invalid_layout` responses, so recovered Markdown still counts as a format failure. Replay records
must agree on request/output IDs and pipeline names; wrong-document records are
rejected. The task requires a single filtered response per document and performs
no reasoning removal or whitespace stripping.

## Scoring

Scoring is pinned to `parse-bench==2.0.1`, dataset revision
`a12c9785cd8cd015b5284e0ad0d74722091762f2`. YAML metadata records those versions,
the prompt version, and renderer version (task version 1.3). Each document is
scored as `parse_bench evaluation run --next_only` scores it; no metric formulas,
eligibility rules or weights are reimplemented:

- A layout reference is cross-evaluated. A page without layout scores 0 in layout
  grounding, as in the official runner; the layout dimension is not dropped.
- Any other reference that the official `ParseEvaluator.can_evaluate` rejects (no
  rules, no reference Markdown, no metadata metric) is skipped, as the official
  CLI skips it. In the pinned snapshot this is `text/text_sparse__anex`.
- The official `_next_only_result` computes the extended metrics. A document no
  extended metric applies to fails; the official aggregation counts it as 0 next
  to successful documents (`text/text_ocr__code`). A scorer error is a harness
  failure, which the official aggregation leaves out instead of adding a 0.
- `EvaluationRunner.aggregate_metrics` and `combine_next_score` give the subset
  averages, six dimensions and overall.

Skipped and failed documents are logged with a warning. `parsebench2_text`,
`parsebench2_formatting`, `parsebench2_tables`, `parsebench2_charts`,
`parsebench2_layout`, and `parsebench2_forms` are the official dimension scores;
`parsebench2_overall` is the official mean of all six. A dimension whose subsets
are absent (for example with `--limit`) is `NaN`, and so is the overall score.
Scores are on [0,1]. `parsebench2_response_failure_rate` and
`parsebench2_layout_coverage` (layout documents with a valid layout channel) are
integration diagnostics, not official dimensions.

## Verification

```bash
uv run --extra parsebench2 python -m pytest test/tasks/test_parsebench2.py test/tasks/test_parsebench2_eligibility.py -q
uv run --extra parsebench2 python scripts/validation/parsebench2_replay.py \
  --per-subset 10 --output /tmp/pb2-replay
```

The regular test suite is offline; add `LMMS_PARSEBENCH2_INTEGRATION=1` for
regressions on pinned public layout documents and on the two text references above.
The replay helper downloads `azure_di_layout` outputs pinned to revision
`80df59e43b88c4474a1c9b90d0315edc70b2b1f8`, renders the selected documents, runs
task scoring, and compares the entire combined report (every slot and document
count, absolute tolerance `1e-12`) and the successful/failed/skipped counts per
subset with the official CLI. It writes a receipt and a `from_log` input, and
prints the matching `lmms_eval --model from_log` command. A large `--per-subset`
covers the whole manifest; rendering every document then takes hours on CPU.

On all 4,317 published `azure_di_layout` records, task scoring and the official CLI
agree on every subset count (4,315 successful, one failed, one skipped) and on the
full report: overall 0.6608, text 0.7968, formatting 0.7766, tables 0.7758, charts
0.0149, layout 0.8380, forms 0.7628.

A fresh run of `Qwen/Qwen2.5-VL-3B-Instruct` (`huggingface` chat backend, one H100,
default generation settings, `--limit 13`) completed all 13 documents with overall
0.243 and response failure rate 0.692: seven responses were invalid JSON (three
reached the 16,384-token limit) and two had layout boxes outside the contract.
JSON conformity, not only transcription quality, limits small models under this
prompt.

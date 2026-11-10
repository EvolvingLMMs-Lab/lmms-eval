# olmOCR-Bench

olmOCR-Bench evaluates document OCR and page-to-Markdown conversion across 1,403 single-page PDF
documents and 8,413 executable unit tests spanning scientific math, historical scans, complex
tables, multi-column reading order, header/footer suppression, and dense tiny text.

- Paper: [olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models (arXiv:2502.18443)](https://arxiv.org/abs/2502.18443)
- Dataset: [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
  at revision `54a96a6fb6a2bd3b297e59869491db4d3625b711`
  (`1,403` PDF pages, `7,019` category unit tests + `1,394` baseline quality checks = `8,413` unit tests).
- Official code: [allenai/olmocr (`olmocr/bench`)](https://github.com/allenai/olmocr/tree/f7cfe4c22098b154c76b6ec950d1c0a464eecf8d/olmocr/bench)
  at commit `f7cfe4c22098b154c76b6ec950d1c0a464eecf8d`.

## Task & Metrics

`olmocr_bench_process_docs` groups the 8,413 unit tests by PDF page into 1,403 page-level evaluation
samples (interleaved round-robin across the 7 source categories so `--limit 7` smoke runs exercise
every category). Each page is rendered to a PNG image (`target_longest_image_dim=2048`) and transcribed
once by the VLM, after which all unit tests associated with that page are evaluated against the single
Markdown output.

| Metric | Category / Source | Official Test Count | Description |
|---|---|---:|---|
| `olmocr_bench_overall` | Macro-average across 8 categories | 8,413 | Unweighted mean of the 8 category pass rates |
| `olmocr_bench_arxiv_math` | `arxiv_math.jsonl` | 2,927 | LaTeX math equations on arXiv pages (`522` PDFs) |
| `olmocr_bench_old_scans_math` | `old_scans_math.jsonl` | 458 | Mathematical formulas in historical scans (`36` PDFs) |
| `olmocr_bench_table_tests` | `table_tests.jsonl` | 1,022 | Cell adjacency (`up`/`down`/`left`/`right`) and heading relations in Markdown/HTML tables (`188` PDFs) |
| `olmocr_bench_old_scans` | `old_scans.jsonl` | 526 | Text presence, absence, and reading order on historical scans (`98` PDFs) |
| `olmocr_bench_headers_footers` | `headers_footers.jsonl` | 760 | Suppression of running headers/footers (`absent` checks on `first_n`/`last_n` characters, `266` PDFs) |
| `olmocr_bench_multi_column` | `multi_column.jsonl` | 884 | Multi-column reading-order verification (`order` checks, `231` PDFs) |
| `olmocr_bench_long_tiny_text` | `long_tiny_text.jsonl` | 442 | Dense small-font text presence checks (`62` PDFs) |
| `olmocr_bench_baseline` | Auto-injected `BaselineTest` | 1,394 | Non-empty alphanumeric output, no >30 trailing n-gram repeats, and no CJK/emoji hallucinations (`1,394` PDFs) |

## Protocol

- **PDF rendering**: Uses the official Poppler path (`pdfinfo` and `pdftoppm`) from `olmocr/data/renderpdf.py`, with `target_longest_image_dim=2048` at `2048 * 72 / max(width_pt, height_pt)` DPI.
- **Prompt**: Defaults to `build_openai_silver_data_prompt_no_document_anchoring` from `olmocr/bench/prompts.py` (`prompt_template: full_no_document_anchoring`), and supports `basic` and `yaml` via `lmms_eval_specific_kwargs`.
- **Scoring**: Uses `TextPresenceTest` (`present`/`absent`), `TextOrderTest` (`order`), `TableTest` (`table`), `MathTest` (`math`), and `BaselineTest` (`baseline`) from the pinned official evaluator. Math comparison renders equations with the pinned KaTeX `0.16.21` assets in Chromium and compares MathML and span geometry.
- **Response boundary**: Scores the returned text without task-specific stripping of reasoning blocks, Markdown fences, YAML front matter, JSON wrappers, or the literal `null`. The task disables framework reasoning-tag removal. A model using a structured output format needs a separately specified conversion protocol; selecting the YAML prompt alone does not enable such a converter.
- **Generation**: The task uses greedy generation with `max_new_tokens=4096` and no text stop sequences. This is the adaptation's configured limit, not a guarantee that every page fits. Model, prompt, image processing, generation budget and response conversion can change scores; arbitrary configurations are not automatically comparable to official leaderboard results.

## Setup

Install the task dependencies and the Chromium browser used by the math scorer:

```bash
uv sync --extra olmocr_bench
uv run --extra olmocr_bench python -m playwright install chromium
```

Install Poppler separately so both `pdfinfo` and `pdftoppm` are on `PATH`:

```bash
# Debian / Ubuntu
sudo apt-get install poppler-utils
# macOS
brew install poppler
```

On Linux, Chromium may also require system libraries; Playwright provides
`python -m playwright install-deps chromium` to install them. The task does not install
packages or browsers during evaluation. Missing scorer dependencies, Chromium or Poppler produce
an actionable error rather than a zero score.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks olmocr_bench \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

## Tests

The default test run requires neither Chromium nor Poppler:

```bash
uv run pytest test/tasks/test_olmocr_bench.py -q
```

After setup, opt into the real PDF and KaTeX rendering checks:

```bash
LMMS_OLMOCR_BENCH_INTEGRATION=1 uv run --extra olmocr_bench pytest test/tasks/test_olmocr_bench.py -q
```

Integration checks use the actual renderers. An explicitly enabled integration run fails if its
required dependencies are unavailable.

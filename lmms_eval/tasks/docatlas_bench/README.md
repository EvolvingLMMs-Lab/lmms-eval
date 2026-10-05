# DocAtlas-Bench

Multilingual document parsing benchmark from
[DocAtlas: Multilingual Document Understanding Across 80+ Languages](https://arxiv.org/abs/2605.12623).
The model converts each page image to Markdown, which is scored against the ground
truth for text, tables and reading order.

- Dataset: [`ahmedheakl/DocAtlas-Bench`](https://huggingface.co/datasets/ahmedheakl/DocAtlas-Bench), `test` split, 5,575 pages
- Official evaluation code: https://github.com/ahmedheakl/DocAtlas/tree/main/eval

## Usage

```bash
pip install beautifulsoup4 func-timeout lxml pylatexenc scipy
lmms-eval --model <model> --tasks docatlas_bench --batch_size 1
```

## Metrics

| Metric | Meaning |
|---|---|
| `docatlas_text_edit` ↓ | normalized edit distance of the text blocks |
| `docatlas_table_teds` ↑ | TEDS of the tables (0–100) |
| `docatlas_reading_order_edit` ↓ | edit distance of the reading order |
| `docatlas_overall` ↑ | mean of text accuracy `100 × (1 − text_edit)` and table TEDS |

The `text_block` and `table` elements of the ground truth are scored. Text and
table scores are averaged per language and then over languages, reading order is
averaged over pages. On the same predictions, the scores equal those of the
official script.

Tables should be predicted as HTML. Markdown pipe tables are converted, LaTeX
tables score 0.

Scoring runs on the CPU, one page at a time, and takes 15 to 80 minutes on the
full split, depending on the model's outputs.

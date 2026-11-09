# DocVQA 2026

DocVQA 2026 evaluates multi-page and high-resolution document visual question answering across
eight domains from the ICDAR 2026 Competition on Document Visual Question Answering.

- Challenge: [ICDAR 2026 DocVQA Challenge (RRC ch=34)](https://rrc.cvc.uab.es/?ch=34)
- Dataset: [VLR-CVC/DocVQA-2026](https://huggingface.co/datasets/VLR-CVC/DocVQA-2026)
  (`val`: 25 documents / 80 questions; `test`: 48 documents / 160 questions with private RRC GT)
- Code: [VLR-CVC/DocVQA2026](https://github.com/VLR-CVC/DocVQA2026)

## Task & Metrics

Each document in `VLR-CVC/DocVQA-2026` contains an ordered list of page images (`1` to `181` pages)
and multiple question-answer pairs. `docvqa_2026_process_docs` flattens the 25 validation documents
into 80 question-level samples (`10` questions per domain).

| Metric | Description |
|---|---|
| `docvqa_2026_acc` | Overall accuracy across all 80 validation questions |
| `docvqa_2026_business_report` | Business reports domain accuracy |
| `docvqa_2026_comics` | Comics domain accuracy |
| `docvqa_2026_engineering_drawing` | Engineering drawings domain accuracy |
| `docvqa_2026_infographics` | Infographics domain accuracy |
| `docvqa_2026_maps` | Maps domain accuracy |
| `docvqa_2026_science_paper` | Science papers domain accuracy |
| `docvqa_2026_science_poster` | Science posters domain accuracy |
| `docvqa_2026_slide` | Presentation slides domain accuracy |

## Protocol

- **Prompt**: Uses the official `MASTER_PROMPT` from `eval_utils.py:get_evaluation_prompt()`, which
  enforces standardized units, ISO dates (`YYYY-MM-DD`), decimal periods, no thousands separators,
  `"Unknown"` for unanswerable questions, and a final line `FINAL ANSWER: <answer>`.
- **Scorer**: Implements `evaluate_docvqa_prediction` from `eval_utils.py`:
  1. Requires `"FINAL ANSWER:"` in the response and extracts the text after the last
     `"FINAL ANSWER:"`.
  2. Parses list-formatted ground truths with `ast.literal_eval`.
  3. **Step A (Strict Match)**: Compares numeric magnitude + unit string (`parse_magnitude_unit`)
     or calendar dates (`dateutil.parser.parse(..., fuzzy=False).date()`, keeping `X.Y.Z` version
     strings literal). Numeric ground truths that fail Step A do not fall back to string
     similarity.
  4. **Step B (Relaxed Text Match)**: Removes punctuation and standalone English articles (`a`,
     `an`, `the`), normalizes whitespace, and accepts answers with normalized Levenshtein
     similarity `>= 0.9`.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks docvqa_2026 \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

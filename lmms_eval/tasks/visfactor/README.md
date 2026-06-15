# VisFactor

[VisFactor](https://github.com/CUHK-ARISE/VisFactor) is a benchmark derived
from the Factor-Referenced Cognitive Test (FRCT) that digitizes 20
vision-centric subtests from established cognitive psychology assessments
to probe the visual-cognition abilities of MLLMs.

* **Paper / repo**: <https://github.com/CUHK-ARISE/VisFactor>
* **Original source data**: `VisFactor.tsv` from
  `https://opencompass.openxlab.space/utils/VLMEval/VisFactor.tsv`
* **HF mirror used here**:
  [`lmms-lab-encoder/visfactor`](https://huggingface.co/datasets/lmms-lab-encoder/visfactor)
  (parquet conversion of the official tsv; `image` is stored as a list of
  `{bytes, path}` structs).

## Tasks

| Task        | Split | Metric            |
| ----------- | ----- | ----------------- |
| `visfactor` | test  | `visfactor_score` |

## Scoring

Ported verbatim from the official VLMEvalKit evaluator
(`vlmeval/dataset/visfactor.py`):

1. Each row's prediction is judged against its `(category_id, additional)`
   pair using the official extraction rules (JSON `{"answer": ...}`,
   followed by category-specific normalisation: boolean / string-list /
   number / coordinate / uppercase-letter).
2. A *test item* is `(category_id, eval_index)` and may span multiple rows;
   an item is correct iff **all** its rows are correct.
3. Per-category accuracy is the fraction of correct items.
4. `visfactor_score` is the macro-average over the 20 categories.

## Usage

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \
    --tasks visfactor \
    --batch_size 1 \
    --output_path ./logs/
```

## Citation

```bibtex
@article{visfactor2025,
  title  = {VisFactor: Benchmarking Multimodal Large Language Models on
            Human Visual Cognition},
  author = {CUHK-ARISE},
  year   = {2025},
  url    = {https://github.com/CUHK-ARISE/VisFactor}
}
```

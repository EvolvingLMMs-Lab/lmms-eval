# MAVIS

[MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](https://arxiv.org/abs/2511.12142) (AAAI 2026) · [code](https://github.com/seokwon99/MAVIS) · [data](https://huggingface.co/datasets/seokwon99/MAVIS)

Given an image, a question about it and a set of text and image documents, the model writes a long-form answer and cites the documents that support each sentence (`[1][2]`). MAVIS has 1,000 human-annotated test answers with atomic facts and supporting documents.

## Task

| Task | Setting | Samples |
| --- | --- | --- |
| `mavis_gt` | GT-RAG: the human-annotated supporting documents (on average 2.8 image and 1.8 text documents) are given in the fixed random order used in the paper | 1,000 |
| `mavis_gt_mini` | The same setting on a fixed 200-answer subset with the domain mix of the full set, for cheaper runs | 200 |

The retrieval settings of the paper (Multi-/Text-/Image-RAG over the 2.5M-document corpus) need the full corpus and are not included; use the [reference code](https://github.com/seokwon99/MAVIS) for them.

## Metrics

An LLM judge scores every answer, with the prompts and scoring rules of the reference code:

| Metric | Meaning |
| --- | --- |
| `mavis_groundedness` | Per-answer F1 of `mavis_recall` and `mavis_precision` |
| `mavis_recall` | Whether each sentence is supported by the documents it cites (uncited sentences score 0) |
| `mavis_precision` | Whether each cited document supports its sentence |
| `mavis_informativeness` | Per-answer F1 of `mavis_completeness` and `mavis_relevance` |
| `mavis_completeness` | How fully the answer covers each annotated atomic fact |
| `mavis_relevance` | Whether each sentence addresses the question and image |

All scores are on a 0–100 scale. Fluency (MAUVE) from the paper is not computed. An empty answer scores 0 on every metric.

## Usage

The paper uses GPT-4.1 as the judge. Each answer takes one judge call per cited sentence, one per citation of sentences that cite several documents, one per sentence for relevance and one per atomic fact (about 35 calls for an 8-sentence answer).

```bash
export OPENAI_API_KEY=...
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks mavis_gt --batch_size 1
```

Judge settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAVIS_JUDGE_API_TYPE` | `openai` | Provider in `lmms_eval.llm_judge` (`openai`, `azure`, `local`, ...) |
| `MAVIS_JUDGE_MODEL` | `gpt-4.1` | Judge model |
| `MAVIS_JUDGE_WORKERS` | `16` | Concurrent judge requests per answer |

The judge must accept images: groundedness on image documents and relevance are judged from the images.

## Citation

```bibtex
@inproceedings{song2026mavis,
  title     = {{MAVIS}: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering},
  author    = {Song, Seokwon and Park, Minsu and Kim, Gunhee},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {39},
  pages     = {33028--33037},
  year      = {2026}
}
```

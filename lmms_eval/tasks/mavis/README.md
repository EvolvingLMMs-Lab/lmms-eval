# MAVIS

[MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](https://arxiv.org/abs/2511.12142) (AAAI 2026) · [code](https://github.com/seokwon99/MAVIS) · [data](https://huggingface.co/datasets/seokwon99/MAVIS)

Given an image, a question about it and a set of text and image documents, the model writes a long-form answer and cites the documents that support each sentence (`[1][2]`). MAVIS has 1,000 human-annotated test answers with atomic facts and supporting documents.

## Tasks

| Task | Setting | Samples |
| --- | --- | --- |
| `mavis_gt` | GT-RAG (gold documents) of the reference code: the human-annotated supporting documents (on average 2.8 image and 1.8 text documents) are given in a fixed random order (seed 0 of the reference code) | 1,000 |
| `mavis_multirag` | Multi-RAG: the top-5 text and image documents retrieved from the corpus (on average 3.0 text and 2.0 image documents) | 1,000 |
| `mavis_textrag` | Text-RAG: the top-5 retrieved text documents | 1,000 |
| `mavis_imagerag` | Image-RAG: the top-5 retrieved image documents | 1,000 |
| `mavis_gt_mini`, `mavis_multirag_mini`, `mavis_textrag_mini`, `mavis_imagerag_mini` | The same settings on a fixed 200-answer subset with the domain mix of the full set (the same answers in every setting), for cheaper runs | 200 |

The retrieved documents are the released retrieval results of the reference code (MM-Embed over the 2.5M-document corpus with one GPT-4o query per question, `data/retrieval/results/gpt-4o_1`), in the same fixed random order. They do not match the documents behind the paper's retrieval tables exactly, so these scores are not directly comparable to the paper's.

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

The judge sends images with `detail: low`, as the reference code does. The paper also generated GPT-4o answers with `detail: low`; to match it with the OpenAI backend, add `image_detail=low` to `--model_args`.

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

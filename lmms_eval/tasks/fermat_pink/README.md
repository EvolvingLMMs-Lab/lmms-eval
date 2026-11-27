# FERMAT-PINK

Faithfulness-aware evaluation of **multi-line handwritten math OCR**, from
*When VLMs ‘Fix’ Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR* (Seong et al., EMNLP 2026).

When VLMs transcribe a student's handwritten solution, they often "fix" the student's mistakes instead of copying them. This is called **over-correction**. The transcription looks fine to string metrics, but the error a teacher needed to see is gone.

**PINK** measures it. An LLM judge grades the VLM transcription and the oracle transcription (what the student actually wrote) with a 5-component, 100-point rubric. Any component on which the transcription scores higher than the oracle is reverted to the oracle score (gain ≤ 10) or set to 0 (gain > 10). Then PINK = Σ penalized / Σ oracle.

- Data: [FERMAT](https://huggingface.co/datasets/ai4bharat/FERMAT) (Nath et al., ACL 2025), 2,244 handwritten solutions with known student errors, CC-BY-4.0. The dataset is gated: accept its terms on the Hub and log in.
- Scorer: [`pink-eval`](https://github.com/9115jin/PINK) (`pip install pink-eval`)

## Metrics

| metric | meaning |
|---|---|
| `pink` | PINK score (higher is better, 1.0 = as faithful as the oracle) |
| `over_correction_rate` | share of samples with at least one over-corrected rubric component |
| `raw_score_ratio` | judge score of the transcriptions / judge score of the oracle, without the penalty |

## Usage

```bash
pip install pink-eval
export OPENAI_API_KEY=...            # default judge: GPT-5 (as in the paper)
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks fermat_pink
```

With the default `gpt-5` judge, the transcriptions are scored against the paper's GPT-5 oracle grades, which ship with `pink-eval`. The results are therefore comparable to the paper's leaderboard.

To use the open-source judge instead, serve it (see the PINK repository) and set the variables below. If no oracle grades are bundled for a judge, the oracle transcriptions are graded on the fly.

```bash
export PINK_JUDGE=qwen3-next-80b PINK_JUDGE_BASE_URL=http://localhost:8000/v1
```

| variable | default | |
|---|---|---|
| `PINK_JUDGE` | `gpt-5` | judge preset (`gpt-5`, `qwen3-next-80b`) |
| `PINK_JUDGE_BASE_URL` | OpenAI API | OpenAI-compatible endpoint of the judge |
| `PINK_JUDGE_MODEL` | from preset | judge model name |
| `PINK_JUDGE_WORKERS` | 16 | parallel judge requests |
| `PINK_ORACLE_GRADES` | bundled grades, if any | oracle grade file for the chosen judge |

## Citation

```bibtex
@inproceedings{seong-etal-2026-vlms,
    title = "When {VLM}s `Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math {OCR}",
    author = "Seong, Jin and Liermann, Wencke and Kim, Minho and Shin, Jong-hun and Lim, Soojong",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.476/",
    pages = "10932--10953"
}

@inproceedings{nath-etal-2025-vision,
    title = "Can Vision-Language Models Evaluate Handwritten Math?",
    author = "Nath, Oikantik and Bathina, Hanani and Khan, Mohammed Safi Ur Rahman and Khapra, Mitesh M",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    url = "https://aclanthology.org/2025.acl-long.720/",
    pages = "14784--14814"
}
```

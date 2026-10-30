# TempCore

## Task Description

[TempCore](https://arxiv.org/abs/2509.01167) (EMNLP 2026) is the temporally
sensitive subset of six Video QA benchmarks: the questions that are both
vision-dependent and whose answer changes with which frames a model sees. A
question is kept when its Language Independence Score (LIS, from no-frame
predictions) is above 0.5 and its Frame Selection Sensitivity (FSS, the
accuracy change between the most and least query-relevant frames, averaged
over eight VLMs) has magnitude above 0.15.

| Task | Source task | Questions |
|------|-------------|----------:|
| `tempcore_mvbench` (group of 18 subtasks) | `mvbench_*` | 209 |
| `tempcore_nextqa` | `nextqa_mc_test` | 629 |
| `tempcore_egoschema` | `egoschema_subset` | 96 |
| `tempcore_mlvu` | `mlvu_dev` | 673 |
| `tempcore_longvideobench` | `longvideobench_val_v` | 261 |
| `tempcore_videomme` | `videomme` | 692 |

Each task includes its source task and keeps only the questions listed in
`tempcore_ids.json` (`process_docs`). Data download, prompts, frame sampling
and scoring are those of the source task, so results are directly comparable
with the full-benchmark score of the same task.

The ids are keyed on fields of the source datasets: MVBench `<task>/<index>`,
NExT-QA `<video>/<qid>`, EgoSchema `video_idx`, MLVU `<task_type>/<question_id>`,
LongVideoBench `id`, Video-MME `question_id`.

## Usage

```bash
python -m lmms_eval --model qwen3_vl --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct \
    --tasks tempcore --batch_size 1
```

Per-sample FSS/LIS scores and the videos are on the Hub:
[`HJOK/TempCore`](https://huggingface.co/datasets/HJOK/TempCore).

## Citation

```bibtex
@inproceedings{ok2026tempcore,
  title     = {TempCore: Are Video QA Benchmarks Temporally Grounded?},
  author    = {Ok, Hyunjong and Lee, Jaeho},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

# EgoLifeQA

EgoLifeQA is a long-context, life-oriented video QA benchmark from the EgoLife project (CVPR 2025).

- Paper: https://arxiv.org/abs/2503.03803
- Project: https://egolife-ai.github.io/
- Dataset: https://huggingface.co/datasets/lmms-lab/EgoLife (EgoLifeQA/EgoLifeQA_A1_JAKE.json, 500 4-way multiple-choice questions)
- Videos: 300 hours of egocentric video under `A1_JAKE/DAY*/` in the same HF repo (30s clips, ~32k files)

## Task

- Modality: Video
- Output type: `multiple_choice` (loglikelihood-based, 4 options A-D)
- Question types: EntityLog, etc. covering recall, health, recommendations over long temporal context.

## Usage

```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks egolifeqa --batch_size 1
```

Videos are resolved via `lmms_eval/tasks/_task_utils/media_resolver.py`:
- Set `EGOLIFEQA_VIDEO_DIR=/path/to/egolife/videos` to use a local copy, or leave unset to use `~/.cache/huggingface/egolifeqa`.
- The `data_files` entry uses `hf://datasets/lmms-lab/EgoLife` so `datasets` will stream/download the QA JSON directly from the Hub without a separate manual step.

## Notes

- Currently one participant (A1_JAKE) is released; the task will automatically pick up additional `EgoLifeQA_*.json` files when they appear by updating `data_files` to a glob.
- For long-context evaluation, `doc_to_visual` returns the single 30s clip nearest to `query_time`; retrieval-augmented methods (EgoRAG) can override this by feeding additional context via the model's retrieval path.

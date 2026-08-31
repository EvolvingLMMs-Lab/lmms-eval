# HERBench

HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering (CVPR 2026).

- Paper: https://arxiv.org/abs/2512.14870
- Project page: https://herbench.github.io/
- Dataset: https://huggingface.co/datasets/DanBenAmi/HERBench
- Code: https://github.com/DanBenAmi/HERBench

HERBench evaluates how vision-language models integrate multiple pieces of evidence in
long videos. Every question is a five-way multiple-choice question that requires
aggregating at least **3 distinct, temporally separated visual cues** (mean Minimum
Required Frame-Set ~5.5), covering 12 compositional task types over 335 videos
(avg. ~6.6 minutes).

## Tasks

| Task | HF config | Questions | Videos | Video download |
|---|---|---|---|---|
| `herbench_full` | `full` | ~27.6k | 335 | ~161 GB |
| `herbench_lite` | `lite` | 2,000 | 68 | ~35 GB |
| `herbench_lite_v2` | `lite_v2` (refined) | 1,971 | 68 | ~35 GB |

`herbench_lite_v2` is the recommended smaller split: 9 of the 12 tasks were regenerated
with additional manual refinement.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=16 \
    --tasks herbench_lite_v2 \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
```

On the first run the videos are downloaded from HuggingFace and extracted to
`$HF_HOME/herbench/videos/`. The download is variant-aware: the lite tasks fetch only
the 4 archive chunks (~35 GB) that contain the 68 lite videos, while `herbench_full`
fetches all 17 chunks (~161 GB). Downloads resume if interrupted; extraction streams
directly from the split chunks (no intermediate concatenated tar), and needs roughly
the same amount of free disk again for the extracted videos.

If you already have the extracted videos elsewhere, point the tasks at them with:

```bash
export HERBENCH_VIDEO_DIR=/path/to/HERBench   # the dir containing the videos/ folder
```

## Metrics

- `herbench_overall_accuracy`: micro-averaged accuracy (%) over all questions
  (random baseline: 20%).
- `herbench_<task_type>_accuracy`: per-task-type accuracy (%) for each of the 12
  compositional task types.

Answers are extracted with the official HERBench letter-extraction protocol
(with the shared lmms-eval MCQ extractor as fallback) and scored by exact match
against the ground-truth letter.

## Citation

```bibtex
@article{herbench2025,
  title={HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering},
  author={Ben-Ami, Dan and Serussi, Gabriele and Cohen, Kobi and Baskin, Chaim},
  journal={arXiv preprint arXiv:2512.14870},
  year={2025}
}
```

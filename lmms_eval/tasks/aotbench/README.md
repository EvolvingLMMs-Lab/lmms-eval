# AoTBench

AoTBench measures whether video models are sensitive to temporal direction. It comes from
[Seeing the Arrow of Time in Large Multimodal Models](https://arxiv.org/abs/2506.03340) (NeurIPS 2025).
Code: [zihuixue/seeAoT](https://github.com/zihuixue/seeAoT). Data:
[sherryxzh/AoTBench](https://huggingface.co/datasets/sherryxzh/AoTBench) (MIT).

| Task | Questions | What it tests |
| --- | ---: | --- |
| `aotbench_reversefilm` | 144 | Which of two segments is played in reverse |
| `aotbench_ucf101` | 500 | Which of two segments is played in reverse |
| `aotbench_rtime_t2v` | 1992 | Which segment matches a directional caption |
| `aotbench_rtime_v2t` | 1992 | Which caption matches the video |
| `aotbench_qa` | 1800 | Temporally demanding questions from nine video QA benchmarks |

The group `aotbench` runs all five tasks. It reports each task separately and has no overall score,
because the official code does not define one.

## Scoring

`aotbench_acc` is accuracy from 0 to 1. As in the official `eval/read_qa.py`, the first character of
the reply is compared with the answer letter, so a reply that does not start with the letter is
wrong. `aotbench_qa` also logs accuracy for each source benchmark.

Generation stops after 16 tokens rather than the official 1024. Only the first character is scored,
so this does not change the result.

## Frame sampling

The task passes the video file to the model, so the frame count is a model argument. The official
script samples 16 frames uniformly. For Qwen2.5-VL:

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,video_nframes=16,max_num_frames=16,attn_implementation=sdpa \
  --tasks aotbench \
  --batch_size 1
```

The paper's settings differ from the official script: 1 FPS with at most 16 frames for Qwen models,
with separate settings for TVBench and Vinoground. Scores from a single setting may therefore not
match every number in the paper.

The first run downloads `videos.zip` (13.5 GB) and extracts it to `$HF_HOME/aotbench`.

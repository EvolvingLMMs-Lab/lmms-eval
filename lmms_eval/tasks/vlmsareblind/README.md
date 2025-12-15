# Vision Language Models Are Blind

## Overview
This benchmark evaluates whether Vision-Language Models (VLMs) can perform simple low-level visual perception tasks that are trivial for humans but require some spatial reasoning. The dataset, called BlindTest, consists of synthetic images designed to probe basic geometric understanding, such as detecting overlaps, intersections, encircled characters, and small object counts. Despite their simplicity, these tasks reveal failures in state-of-the-art VLMs (SOTA as of June 2024), and highlight limitations in fine-grained visual perception.

## Paper Information

- **Paper Title**: Vision Language Models Are Blind
- **Paper**: https://arxiv.org/abs/2407.06581
- **GitHub**: https://github.com/xai-org/vlmsareblind
- **Dataset**: https://huggingface.co/datasets/XAI/vlmsareblind

## Dataset Details

The benchmark consists of 9 different tasks testing different visual reasoning capabilities:

| Task | Description |
|------|-------------|
| **Circled Letter** | Identify which letter is circled |
| **Counting Grid** | Count rows and columns in blank and word grids |
| **Line Plot Intersections** | Count line intersections |
| **Nested Squares** | Count total squares in nested pattern |
| **Olympic Counting - Circles** | Count circles (Olympic rings style) |
| **Olympic Counting - Pentagons** | Count pentagons |
| **Subway Connections** | Count single-color paths between stations |
| **Touching Circles** | Determine if two circles touch |

**Note:** This implementation follows the paper's grouping of tasks as displayed in Table 1, i.e. the counting grid tasks (`Counting Grid - Blank Grids` and `Counting Grid - Word Grids`) are grouped into a single task (`Counting Grid`). 

## Dataset Fields

The benchmark uses the following fields from the huggingface dataset:
- `task`: The task type (one of the 8 types above)
- `image`: The visual input
- `prompt`: The question with answer format instructions
- `groundtruth`: The correct answer

## Metrics
- **accuracy**: Mean accuracy across all samples (micro-averaged overall fraction correct).
- **accuracy_by_task**: Per-task accuracy. Includes `task_mean`, the macro-average (unweighted mean) of per-task accuracies.

## Usage

```bash
uv run -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen3-VL-2B-Instruct \
  --tasks vlmsareblind \
  --batch_size 1 \
  --device cuda:0
```

## Expected Results
Below are example results for several Qwen3-VL models plus the original-paper baselines (italicized).

| Model | Line Plot Intersections | Touching Circles | Circled Letter | Olympic Counting - Circles | Olympic Counting - Pentagons | Nested Squares | Grid Counting | Subway Connections | Task Mean |
|-------|--------------------------|------------------|----------------|----------------------------|-----------------------------|---------------|--------------|--------------------|-----------|
| _GPT-4o_ | _41.61_ | _75.91_ | _74.23_ | _41.25_ | _20.21_ | _55.83_ | _39.58_ | _53.19_ | ***50.23*** |
| _Gemini-1.5_ | _66.94_ | _93.62_ | _83.29_ | _20.25_ | _24.17_ | _87.08_ | _39.39_ | _53.13_ | ***58.48*** |
| _Sonnet-3_ | _43.41_ | _86.46_ | _72.06_ | _29.79_ | _1.87_ | _65.00_ | _36.17_ | _31.11_ | ***45.73*** |
| _Sonnet-3.5_ | _75.36_ | _90.82_ | _87.88_ | _66.46_ | _77.71_ | _92.08_ | _74.26_ | _58.19_ | ***77.84*** |
| Qwen3-VL-2B-Instruct | 55.08 | 72.54 | 75.32 | 20.21 | 4.58 | 66.67 | 21.21 | 20.28 | **41.99** |
| Qwen3-VL-4B-Instruct | 77.11 | 72.32 | 92.79 | 21.04 | 11.88 | 97.08 | 50.19 | 40.42 | **57.85** |
| Qwen3-VL-8B-Instruct | 72.64 | 88.69 | 86.06 | 20.00 | 12.08 | 89.17 | 66.29 | 40.14 | **59.38** |


## Generation Configuration

- **max_new_tokens**: 32
- **temperature**: 0
- **top_p**: 1.0
- **num_beams**: 1
- **do_sample**: false

## Citation

```bibtex
@InProceedings{Rahmanzadehgervi_2024_ACCV,
    author    = {Rahmanzadehgervi, Pooyan and Bolton, Logan and Taesiri, Mohammad Reza and Nguyen, Anh Totti},
    title     = {Vision language models are blind},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {18-34}
}
```

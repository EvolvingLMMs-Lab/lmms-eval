# XModBench

**XModBench** is a cross-modal multiple-choice question answering benchmark that evaluates multimodal language models across all combinations of Audio, Image, Video, and Text modalities.

Each sample presents:
- A **condition** (the question stimulus) in one modality (Audio, Image, Video, or Text)
- Four **options** (A/B/C/D), each in another modality (Audio, Image, Video, or Text)
- A **question** asking the model to select the best matching option

## Dataset

The benchmark is built from [AudioBench](https://github.com/AudioBench/AudioBench) and covers 5 categories:

| Category | Subtasks |
|---|---|
| Perception | General activities, Fine-grained, Instruments, Instrument composition, Nature sounds |
| Spatial | Arrangements, 3D movements, Panorama |
| Speech | Recognition, Translation |
| Temporal | Count, Calculation, Order |
| External | Music genre, Emotion, Movie matching, Singer identification |

## Modality Combinations

| Task name | Condition | Options | Samples |
|---|---|---|---:|
| `xmod_bench_audio_image` | Audio | Image | 11,689 |
| `xmod_bench_audio_text` | Audio | Text | 14,719 |
| `xmod_bench_audio_video` | Audio | Video | 3,031 |
| `xmod_bench_image_audio` | Image | Audio | 11,689 |
| `xmod_bench_image_text` | Image | Text | 11,689 |
| `xmod_bench_text_audio` | Text | Audio | 14,725 |
| `xmod_bench_text_image` | Text | Image | 11,689 |
| `xmod_bench_text_video` | Text | Video | 3,031 |
| `xmod_bench_video_audio` | Video | Audio | 3,036 |
| `xmod_bench_video_text` | Video | Text | 3,036 |
| **`xmod_bench`** (group) | All | All | **88,334** |

## Setup

### 1. Prepare the data

Set the `AUDIOBENCH_ROOT` environment variable to point to your AudioBench root directory (defaults to `/home/xwang378/scratch/2025/AudioBench`):

```bash
export AUDIOBENCH_ROOT=/path/to/AudioBench
```

The preprocessed JSONL files are stored in `data/` and already embedded in this task directory. They are generated from the raw AudioBench JSON files and have a unified schema.

If you need to regenerate the JSONL files (e.g., after updating the raw data), run:

```bash
python3 -c "
import json, os

TASKS_ROOT = '$AUDIOBENCH_ROOT/benchmark/tasks'
OUT_DIR = 'lmms_eval/tasks/xmod_bench/data'
# ... (see the preprocessing logic in utils.py comments)
"
```

### 2. Run evaluation

**Quick test (limit 8 samples, audio→text only):**
```bash
python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks xmod_bench_audio_text \
    --batch_size 1 \
    --limit 8
```

**Single modality combination:**
```bash
python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks xmod_bench_audio_image \
    --batch_size 1
```

**Full benchmark (all 10 modality combinations):**
```bash
python -m lmms_eval \
    --model qwen2_5_omni \
    --model_args pretrained=Qwen/Qwen2.5-Omni-7B \
    --tasks xmod_bench \
    --batch_size 1
```

## Prompt Format

Each sample is formatted as an interleaved multi-modal message:

```
[condition media]
<question text>
A: [option A media / text]
B: [option B media / text]
C: [option C media / text]
D: [option D media / text]
Answer with the option's letter (A, B, C, or D) directly.
```

For text-modality options, the option content is inlined as `A. <text>`.
For text-modality conditions, the context is prepended as `Context: <text>`.

## Metrics

- **Overall accuracy**: percentage of correct answers across all samples
- **Per modality-combo accuracy**: breakdown by condition→option modality pair (e.g., `audio->image`)
- **Per category accuracy**: breakdown by benchmark subtask (e.g., `01_perception/finegrained`)

Both breakdowns are logged automatically at the end of evaluation.

## File Structure

```
xmod_bench/
├── README.md                      # This file
├── _default_template_yaml         # Shared YAML defaults
├── utils.py                       # Processing functions
├── xmod_bench.yaml                # Group: runs all 10 subtasks
├── xmod_bench_audio_image.yaml
├── xmod_bench_audio_text.yaml
├── xmod_bench_audio_video.yaml
├── xmod_bench_image_audio.yaml
├── xmod_bench_image_text.yaml
├── xmod_bench_text_audio.yaml
├── xmod_bench_text_image.yaml
├── xmod_bench_text_video.yaml
├── xmod_bench_video_audio.yaml
├── xmod_bench_video_text.yaml
└── data/
    ├── audio_image.jsonl          # Normalized, merged JSONL per modality combo
    ├── audio_text.jsonl
    ├── ...
    └── video_text.jsonl
```

## Citation

If you use XModBench, please cite:

```bibtex
@misc{xmodbench2025,
  title   = {XModBench: A Cross-Modal Multiple-Choice Benchmark for Multimodal Language Models},
  author  = {...},
  year    = {2025},
}
```

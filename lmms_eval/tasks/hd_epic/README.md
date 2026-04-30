# HD-EPIC VQA Benchmark

[HD-EPIC](https://hd-epic.github.io/) (A Highly-Detailed Egocentric Video Dataset) is a video question answering benchmark from Perrett et al., CVPR 2025. It covers egocentric kitchen activities with 30 question prototypes across 7 categories, generating 26,550 multiple-choice questions from 41 hours of video.

## Categories and Prototypes

| Category | Prototypes | Questions |
|---|---|---|
| Recipe | Recipe Recognition, Multi-Recipe Recognition, Multi-Step Localisation, Step Localisation, Prep Localisation, Step Recognition, Rough Step Localisation, Following Activity Recognition | 8 |
| Ingredient | Ingredient Retrieval, Ingredient Weight, Ingredients Order, Ingredient Adding Localisation, Ingredient Recognition, Exact Ingredient Recognition | 6 |
| Nutrition | Image Nutrition Estimation, Nutrition Change, Video Nutrition Estimation | 3 |
| Fine-grained Actions | Action Recognition, How Recognition, Why Recognition, Action Localisation | 4 |
| 3D Perception | Fixture Location, Object Location, Object Contents Retrieval, Fixture Interaction Counting | 4 |
| Object Motion | Object Movement Itinerary, Object Movement Counting, Stationary Object Localisation | 3 |
| Gaze | Gaze Estimation, Interaction Anticipation | 2 |

## Setup

### 1. Download videos

Follow the instructions at [hd-epic.github.io](https://hd-epic.github.io/) to download the dataset videos. Videos should be organised as:

```
/path/to/videos/
  P01/
    P01-20240427-151808.mp4
    ...
  P02/
    ...
```

### 2. Download annotations

```bash
git clone https://github.com/hd-epic/hd-epic-annotations
```

### 3. Convert annotations to JSONL

```bash
python lmms_eval/tasks/hd_epic/hd_epic_to_hf.py \
    --questions-dir hd-epic-annotations/vqa-benchmark \
    --output lmms_eval/tasks/hd_epic/hd_epic_questions.jsonl \
    --video-dir /path/to/videos
```

### 4. Set the video directory environment variable

```bash
export HD_EPIC_VIDEO_DIR=/path/to/videos
```

This environment variable overrides the `video_dir` field baked into the JSONL at conversion time, so you can move videos without reconverting.

## Running Evaluations

### Full benchmark (all 30 prototypes)

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks hd_epic \
    --batch_size 1 \
    --output_path ./logs/hd_epic_full
```

### By category

```bash
# e.g. recipe category only
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks hd_epic_recipe \
    --batch_size 1 \
    --output_path ./logs/hd_epic_recipe
```

Available category tasks: `hd_epic_recipe`, `hd_epic_ingredient`, `hd_epic_nutrition`, `hd_epic_fine_grained`, `hd_epic_3d_perception`, `hd_epic_object_motion`, `hd_epic_gaze`.

### Single prototype

```bash
# e.g. gaze estimation only
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks hd_epic_gaze_gaze_estimation \
    --batch_size 1 \
    --output_path ./logs/hd_epic_gaze_estimation
```

## Validation Results

Validated using Qwen2.5-VL-7B-Instruct with settings matching the R3 community report (Zhang et al., 2025): `fps=1, max_num_frames=32, min_pixels=50176, max_pixels=50176` (224×224 per frame).

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,fps=1,max_num_frames=32,min_pixels=50176,max_pixels=50176 \
    --tasks hd_epic_ingredient_ingredient_weight \
    --batch_size 1 \
    --output_path ./logs/qwen_r3_match
```

| Prototype | R3 report (Zhang et al.) | This integration | Δ |
|---|---|---|---|
| Ingredient Weight | ~28% | 26% | -2pp (within SE) |

The 2pp difference is within the statistical noise for n=50 questions (SE ≈ 6pp). Results are consistent with the published zero-shot Qwen2.5-VL-7B baseline.

**Note on frame sampling:** Higher frame counts or resolution improve accuracy on short-clip prototypes. The default lmms-eval Qwen2.5-VL settings (32 frames, default pixel budget) give ~40% on Ingredient Weight — above R3's 28% because more frames are sampled per second for short clips. To reproduce R3's numbers exactly, use `fps=1,max_num_frames=32,min_pixels=50176,max_pixels=50176`.

## Notes

- **Clip extraction**: `ffmpeg -c copy` is used for fast, lossless trimming to the question's time window. If keyframe alignment causes duration mismatches, replace `-c copy` with `-c:v libx264 -preset ultrafast` in `utils.py:_extract_clip`.
- **BBOX coordinates**: Normalised from native 1408×1408 (Project Aria RGB camera) to 1000×1000, matching the official HD-EPIC eval protocol.
- **TIME tags**: Made relative to clip start, consistent with the original `hd-epic-vqa-eval` repository.
- **Multi-video questions**: Videos are passed in order as separate `{"type": "video", "url": ...}` blocks in the user message.
- **JSONL file**: `hd_epic_questions.jsonl` is generated locally and should not be committed to the repository (it is listed in `.gitignore`).

## Citation

```bibtex
@inproceedings{perrett2025hdepic,
  title     = {{HD-EPIC}: A Highly-Detailed Egocentric Video Dataset},
  author    = {Perrett, Toby and Darkhalil, Ahmad and Sinha, Saptarshi and
               Emara, Omar and Pollard, Sam and Parida, Kranti and Liu, Kaiting and
               Gatti, Prajwal and Bansal, Siddhant and Flanagan, Kevin and
               Chalk, Jacob and Zhu, Zhifan and Guerrier, Rhodri and
               Abdelazim, Fahd and Zhu, Bin and Moltisanti, Davide and
               Wray, Michael and Doughty, Hazel and Damen, Dima},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

# Spatial Scene Bench

[Code](https://github.com/selfishout/spatial-scene-bench) | [Dataset](https://huggingface.co/datasets/alithema/spatial-scene-bench)

Spatial Scene Bench probes counting and relational reasoning in procedurally rendered 3D scenes, from single
images to multi-view orbit videos. Every answer is computed from the 3D scene itself, and ambiguous questions
(heavily occluded objects, near-ties in depth or distance) are filtered out at generation time.

| Type | Example | Metric |
| --- | --- | --- |
| `count`, `count_relation` | How many cubes are to the left of the red sphere? | exact match (MRA also reported) |
| `relation`, `compare_count` | Is the small cylinder behind the blue cube? | exact match |
| `closest` | Which object is closest to the gray cube in 3D space? (A)…(D) | exact match |
| `distance` | What is the distance between A and B, in meters? | MRA |
| `ego_direction` | Imagine you are standing at A and facing B. Where is C relative to you? | exact match |
| `video_count`, `video_closest`, `video_distance`, `video_ego_direction` | same skills from 16 orbit frames | as above |

Numeric answers use the Mean Relative Accuracy (MRA) of VSI-Bench.

## Tasks

- `spatial_scene_bench`: full test set, 200 scenes and 4,387 questions.
- `spatial_scene_bench_mini`: the first 25 questions of each type (275 questions).

## Metrics

- `ssb_overall`: macro average of per-type accuracy (MRA for `distance` and `video_distance`).
- `ssb_image_overall`, `ssb_video_overall`: the same average over single-image or video types.
- `ssb_count_mra`: MRA over all counting questions, which rewards near misses.
- `<type>_accuracy`: per question type.

## Usage

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks spatial_scene_bench_mini \
    --batch_size 1
```

# ShareRobot-Bench

Tasks: `share_robot_affordance` (200 box questions), `share_robot_trajectory` (200 trajectory questions), and group `share_robot` (both).

The [Apache-2.0 dataset](https://huggingface.co/datasets/BAAI/ShareRobot-Bench) is loaded from the two separate test JSON files to avoid the dataset-root mixed image-field schema. Images are cached through Hugging Face Hub. The trajectory task uses the first image and first conversation pair, matching the reference; prompts retain its instructions. For Qwen2.5-VL affordance evaluation, task scoring converts raw predictions to original-image pixels. Both model adapters remain unchanged; saved responses retain the original model text. The scoring callback receives resolved generation settings through an explicit `generation_kwargs` keyword.

```bash
uv run --no-sync accelerate launch --multi_gpu --num_processes 2 \
  --num_machines 1 --mixed_precision no --dynamo_backend no -m lmms_eval \
  --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=sdpa \
  --tasks share_robot --limit 8 --batch_size 1 --log_samples --output_path outputs/share_robot
```

The limit applies separately to each task. Omit it for the complete benchmark. Use `doc_to_messages` with chat backends or text/visual adapters with simple backends.

## Scoring

Reference: [FlagEvalMM PR #84](https://github.com/flageval-baai/FlagEvalMM/pull/84), commit `24655d281c9d082592942cd18168cca5d0951fdb`.

- Affordance: `average_iou`, `iou_at_0_5`, `iou_at_0_75`, `iou_at_0_9` use the valid-answer denominator. `average_precision` is the reference's compatibility alias for mean IoU, **not detection AP**. Malformed/nonfinite predictions are excluded; reversed or zero-area predictions score zero when compared with valid ground truth. `valid_items` and `valid_rate` expose exclusions. An all-invalid set returns zero metrics and zero validity, matching the reference convention.
- Trajectory: normalized discrete Fréchet, symmetric Hausdorff, and Euclidean RMSE (all lower is better). Literal/list parsing divides both coordinates by image dimensions when either coordinate exceeds one; the reference line-by-line fallback scales axes individually. RMSE compares the shorter trajectory prefix; Fréchet and Hausdorff use all points. Unparseable predictions fall back to the origin and remain in distance averages, with `parse_valid_items`/`parse_valid_rate` exposing that fallback.
- Invalid/missing ground truth and unreadable media fail explicitly. Model output is parsed with safe literals, never executed. No overall score combines the two tasks.

## Explicit overlap metrics

The affordance task additionally reports these scores across **all examples**, counting invalid predictions as failures:

| Metric | Formula | Range |
| --- | --- | --- |
| `mean_iou` | mean of `I / U` | [0, 1] |
| `cumulative_iou` | sum of `I` / sum of `U` across all workers | [0, 1] |
| `mean_generalized_iou` | mean of `I / U - (C - U) / C` | [-1, 1] |
| `precision_at_0_5` | fraction with ordinary IoU >= 0.5 | [0, 1] |

`I` and `U` are the intersection and union areas; `C` is the smallest axis-aligned enclosing rectangle's area. Geometry uses original pixel coordinates and continuous areas (no inclusive-pixel `+1`, clipping, rasterization or coordinate rescaling). Cumulative IoU weights examples by union area, so image resolution affects its weighting. Scores are fractions; multiply by 100 for percentages. Empty aggregation returns zero.

Missing, malformed, nonfinite, reversed or zero-area predictions score zero in the new scalar metrics; cumulative IoU counts zero intersection and the ground-truth area as union. Nonfinite derived geometry is also a prediction failure. Valid disjoint boxes retain their actual union and may score negative generalized IoU. Ground-truth boxes must have finite positive area. Existing reference metrics retain their valid-answer denominators and validity diagnostics; the new and old means can therefore differ.

The names avoid an ambiguity: [LISA](https://arxiv.org/html/2308.00692v3#S5.SS1) uses gIoU for mean per-image IoU and cIoU for cumulative IoU, whereas [SpatialAfford v1](https://arxiv.org/html/2608.00502v1#S4.SS1) calls gIoU generalized IoU without defining cIoU. These explicit metrics cover both gIoU interpretations but do **not** establish exact SpatialAfford evaluator parity. Cumulative IoU is distinct from Complete IoU (which includes center-distance and aspect-ratio penalties); Complete IoU is not reported. Trajectory scoring is unchanged.

## Offline workers and validation

Cache the full Hub repository (about 59 MB of images) and use a shared writable `HF_DATASETS_CACHE`. The packaged task configs use annotation URLs: the datasets/Hub filesystem may still resolve repository metadata even after Arrow caches are warmed. Workers using these original configs need network/proxy access. For fully offline execution, make local task configs pointing `dataset_kwargs.data_files.test` to the corresponding cached JSON file, retaining all prompts and scoring functions, and verify those configs with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` before allocating GPUs. A Hub snapshot alone does not make URL-based task configs offline-capable.

Run `uv run pytest test/eval/test_share_robot.py -q` and the repository pre-commit checks. Real E2E evidence must include both tasks, media loading, non-empty predictions, validity metrics and distance/IoU metrics. Ground-truth and parser fixtures are tested separately from model quality.

Version 0.4 performs conversion entirely in task scoring. `model_specific_generation_kwargs` selects `share_robot_coordinate_profile: qwen2_5_vl_chat` for `qwen2_5_vl` and `qwen2_5_vl_chat`. Other models retain original-coordinate scoring. The chat profile reproduces qwen-vl-utils' initial default resize followed by processor resizing; the `qwen2_5_vl_simple` profile applies configured limits at both stages. Both include rounding to multiples of 28. Installed Qwen/Transformers resize functions are used and tested against actual processor image grids.

The task-owned `share_robot_min_pixels: 200704` and `share_robot_max_pixels: 1605632` describe the processor's expected limits; they do **not** configure the model. If overriding `min_pixels`/`max_pixels` (or `image_min_pixels`/`image_max_pixels`) in model arguments, also set matching `share_robot_min_pixels`/`share_robot_max_pixels` via `--gen_kwargs`. When using `--force_simple`, additionally set `--gen_kwargs share_robot_coordinate_profile=qwen2_5_vl_simple`. Do not assume arbitrary custom preprocessing matches these profiles.

Prompts, decoding, trajectory scoring, and metric formulas are unchanged. Earlier runs compared raw resized-image predictions against original-image annotations and remain historical evidence. Job `pt-x38hw4nj` ran successfully but bypassed the initial simple-adapter-only conversion through the default chat backend; it does not validate that fix. Full-set post-hoc rescoring is distinct from fresh inference. Conversion corrects coordinate space but does not establish exact parity with the SpatialAfford evaluator.

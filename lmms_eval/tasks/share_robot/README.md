# ShareRobot-Bench

Tasks: `share_robot_affordance` (200 box questions), `share_robot_trajectory` (200 trajectory questions), and group `share_robot` (both).

The [Apache-2.0 dataset](https://huggingface.co/datasets/BAAI/ShareRobot-Bench) is loaded from the two separate test JSON files to avoid the dataset-root mixed image-field schema. Images are cached through Hugging Face Hub. The trajectory task uses the first image and first conversation pair, matching the reference; prompts retain its instructions. No model-specific coordinate scaling is added.

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

## Offline workers and validation

Cache the full Hub repository (about 59 MB of images) and use a shared writable `HF_DATASETS_CACHE`. The packaged task configs use annotation URLs: the datasets/Hub filesystem may still resolve repository metadata even after Arrow caches are warmed. Workers using these original configs need network/proxy access. For fully offline execution, make local task configs pointing `dataset_kwargs.data_files.test` to the corresponding cached JSON file, retaining all prompts and scoring functions, and verify those configs with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` before allocating GPUs. A Hub snapshot alone does not make URL-based task configs offline-capable.

Run `uv run pytest test/eval/test_share_robot.py -q` and the repository pre-commit checks. Real E2E evidence must include both tasks, media loading, non-empty predictions, validity metrics and distance/IoU metrics. Ground-truth and parser fixtures are tested separately from model quality.

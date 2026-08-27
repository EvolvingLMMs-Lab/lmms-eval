# Model Benchmark Tags

This repo can already index `tag` values as first-class task selectors. We use that mechanism to attach tasks to model families that publicly highlight the benchmark in official evaluation materials.

## Naming

Use `public_eval_<model_family>` for curated "officially highlighted" benchmark tags.

Current tags:

- `public_eval_gemini3_family`
- `public_eval_gpt5_family`
- `public_eval_qwen3_5_family`
- `public_eval_seed2_family`

These tags are intentionally model-family scoped. For example, `public_eval_gpt5_family` can cover GPT-5-era official materials without creating a new tag for every checkpoint suffix.

## Scope

This first pass covers the verified image benchmark set listed in `AGENTS.md`.

The tags mean:

- The task appears in official public benchmark materials for that model family.
- The task is mapped to the closest runnable `lmms_eval` task.
- The tags are curated, not auto-generated from every community leaderboard mention.

They do not mean:

- The benchmark is unique to that model family.
- The benchmark is the only task worth running for that model family.
- The exact prompt or protocol is guaranteed to be identical to the vendor's internal eval.

## Initial Mapping

- `public_eval_qwen3_5_family`:
  `mmmu_val`, `mmmu_pro_vision`, `mmmu_pro_standard`, `mathvista_testmini_*`, `ai2d`, `mmbench_en_dev`, `ocrbench`, `realworldqa`, `mmstar`, `hallusion_bench_image`, `mmlongbench_doc`, `omnidocbench`, `charxiv_val_reasoning`
- `public_eval_seed2_family`:
  `mathvision_testmini`, `ocrbench_v2`, `mmlongbench_doc`, `omnidocbench`, `charxiv_val_reasoning`, `charxiv_val_descriptive`
- `public_eval_gemini3_family`:
  `mmmu_pro_vision`, `mmmu_pro_standard`, `omnidocbench`, `charxiv_val_reasoning`
- `public_eval_gpt5_family`:
  `mmmu_val`

## Usage

Run all tasks associated with a model family:

```bash
uv run python -m lmms_eval --tasks public_eval_qwen3_5_family
```

Inspect registered tags from Python:

```python
from lmms_eval.tasks import TaskManager

tm = TaskManager()
print(tm.all_tags)
print(tm.task_index["public_eval_qwen3_5_family"]["task"])
```

## Maintenance

When adding a new model-family tag:

1. Prefer official sources over third-party leaderboards.
2. Tag concrete task YAMLs, not group YAMLs.
3. Reuse the closest existing `lmms_eval` task instead of inventing a near-duplicate task name.
4. Keep the mapping conservative when the vendor benchmark protocol is ambiguous.

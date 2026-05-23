# Apertus VLM eval (vLLM + lmms-eval, CSCS)

Self-contained launcher for evaluating Apertus 1.5 VLMs with vLLM on CSCS.
One CLI, one SLURM template, per-task SQLite image-token cache.

## Quickstart

```bash
# Run the full eval suite (~46 tasks) on a checkpoint
bash scripts/eval.sh /path/to/ckpt

# Sanity check (3 tasks, ~5 min/model)
bash scripts/eval.sh /path/to/ckpt --suite smoke

# Specific tasks
bash scripts/eval.sh /path/to/ckpt --tasks mmmu_val,chartqa

# Multiple models
bash scripts/eval.sh /a,/b,/c

# Many models from a file (one path per line, # comments OK)
bash scripts/eval.sh @models.txt

# Cache fill (first run on a new task — populates per-task SQLite)
bash scripts/eval.sh /path/to/ckpt --tasks newtask --mode fill
```

Results land under `results/<model_name>/`, logs under `logs/`.
Use `python scripts/gather_results.py` to aggregate across models.

## What this folder contains

| File | Purpose |
|---|---|
| `scripts/eval.sh` | **User-facing CLI**. The thing you invoke. |
| `scripts/eval_job.slurm` | One SLURM template. Called by `eval.sh` per (model, task). Supports direct invocation for advanced use. |
| `scripts/gather_results.py` | Aggregate per-model `results/.../*_results.json` into a comparison table. |
| `scripts/push_results_to_wandb.py` | Push existing results JSONs to W&B (post-hoc). |
| `apertus-vllm-lmms-eval-prod.toml` | Pyxis container env file (squash image, mounts). |
| `squash/` | Production squash image. |
| `cache/`, `results/`, `responses/`, `logs/` | Outputs (gitignored). |

## CLI reference

```
bash scripts/eval.sh <model> [--tasks T | --suite full|smoke] [--mode fill|readonly]
```

| Arg | Form | Notes |
|---|---|---|
| `<model>` | `/path/to/ckpt` | single checkpoint |
| | `/a,/b,/c` | comma-separated paths |
| | `@models.txt` | one path per line (`#` comments OK) |
| `--tasks` | `mmmu_val,chartqa` | comma-separated task names |
| | `@tasks.txt` | one task per line |
| | _(omitted)_ | uses `--suite` |
| `--suite` | `full` (default) | the curated 46-task working set |
| | `smoke` | `gqa,mmstar,pope` — sanity canary |
| `--mode` | `readonly` (default) | eval against per-task cache (no writes) |
| | `fill` | populate per-task cache (first run on a task) |

### Common env overrides

| Var | Default | Purpose |
|---|---|---|
| `ENABLE_WANDB` | `true` | Push metrics to W&B (`alvor/apertus-1p5-eval` by default) |
| `WANDB_API_KEY` | (auto from `~/.netrc`) | If unset and `~/.netrc` lookup fails, jobs fail fast. |
| `HF_TOKEN` | (auto from `~/.cache/huggingface/token`) | Re-exported into the job for gated datasets. |
| `GEN_KWARGS` | `max_new_tokens=8192,temperature=0` | Lower budgets truncate reasoning tasks. |
| `NUM_PROCESSES` | `4` | vLLM workers per node (1 per GH200 GPU). |
| `CACHE_BASE` | `/capstor/.../benchmark/image_token_cache` | Per-task SQLite root. |
| `LOG_DIR` | `/capstor/.../examples/apertus-vllm/logs` | sbatch stdout/err destination. |

## Per-task SQLite image-token cache

Each task gets its own SQLite under `$CACHE_BASE/<task>/image_tokens/apertus_image_token_cache.sqlite3`.
Bounded growth per file, trivially parallel writes during fill (different files
→ no lock contention), blast-radius per task on corruption.

Cache lifecycle:

```
# First time you eval on a task (populates the per-task cache)
bash scripts/eval.sh /path/to/ckpt --tasks newtask --mode fill

# All subsequent runs read the populated cache, no writes
bash scripts/eval.sh /path/to/ckpt --tasks newtask              # (readonly is the default)
```

Per-task caches across recipe variants (e.g. `VisualPuzzles_direct` vs `VisualPuzzles_cot`)
are stored separately. Image-tokens are small bytes so this duplication is fine
and the per-task isolation is worth it.

## Curating the suite

The `--suite full` list lives as the `SUITE_FULL` constant inside `eval.sh`.
Add/remove tasks by editing that constant — no external file to keep in sync.

To run an ad-hoc subset without editing the suite, use `--tasks t1,t2,t3` or
`--tasks @my_subset.txt` (the file form is `.gitignore`d, so workflow scratch
stays local).

## Advanced: direct `sbatch eval_job.slurm`

The orchestrator is `eval.sh`. For one-off jobs where you want to override
SLURM resources or eval-launcher args directly:

```bash
sbatch scripts/eval_job.slurm \
  --model-path /path/to/ckpt \
  --tokenizer-path /capstor/.../tokenizer/apertus_emu3.5_instruct \
  --tasks mmmu_val \
  --image-token-cache-dir $CACHE_BASE/mmmu_val \
  --image-token-cache-local-copy 0 \
  --image-token-cache-readonly 1 \
  --image-token-cache-write-misses 0
```

`bash scripts/eval_job.slurm --help` lists all flags.

## Production root

`#SBATCH` paths inside `eval_job.slurm` are hardcoded to:

```
/capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm
```

This is the canonical infra-mounted copy. From any user shell, run from your
git working copy: `bash scripts/eval.sh ...`. The orchestrator submits via
sbatch which then reads the hardcoded `#SBATCH` paths to land outputs under
the shared infra root (not your scratch).

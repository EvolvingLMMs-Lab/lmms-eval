# Apertus vLLM lmms-eval Production Launcher

This folder is self-contained for the Apertus + vLLM `lmms-eval` workflow on CSCS.

Production root:

```bash
/capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm
```

The Slurm `#SBATCH` paths are intentionally hardcoded to this folder, because Slurm reads those directives before the script body runs. From a user point of view, the intended workflow is: enter this folder, run the README command, and outputs land back in this folder.

## What Is Included

- `apertus-vllm-lmms-eval-prod.toml`: Slurm environment file pointing at the local `squash/` image.
- `squash/apertus-vllm-lmms-eval-13.0-prod.sqsh`: production squash image used by Slurm.
- `scripts/run_lmms_eval_vllm_eval_local_copy.slurm`: eval launcher using node-local readonly SQLite cache copy.
- `scripts/run_lmms_eval_vllm_fill_shared_cache.slurm`: cache-fill launcher.
- `scripts/run_lmms_eval_vllm_prod.slurm`: generic launcher.
- `scripts/eval_using_local_sqlite_copy.sh`: submits model x task eval jobs from `scripts/models.txt`.
- `scripts/eval_using_read_only_sqlite.sh`: submits model x task eval jobs using shared SQLite directly (no node-local copy, no writes).
- `scripts/fill_shared_sqlite_cache.sh`: submits model x task cache-fill jobs from `scripts/models.txt`.
- `scripts/models.txt`: model paths, one per line.
- `cache/image_tokens/`: copied SQLite image-token cache DB.
- `results/`, `responses/`, `logs/`, `scripts/logs/`: copied benchmark artifacts and logs.

## Quick Start

Go into this folder:

```bash
cd /capstor/store/cscs/swissai/infra01/multimodal-eval/apertus-lmms-eval/lmms-eval/examples/apertus-vllm
```

Submit one eval job using the first model in `scripts/models.txt`:

```bash
MODEL_PATH="$(awk 'NF && $1 !~ /^#/ {print; exit}' scripts/models.txt)"
sbatch scripts/run_lmms_eval_vllm_eval_local_copy.slurm \
  --model-path "${MODEL_PATH}" \
  --tasks realworldqa \
  --enable-wandb false
```

That command uses these folder-local defaults:

- `results/` for lmms-eval outputs
- `logs/` for Slurm logs from direct `sbatch`
- `cache/` for shared caches
- `cache/image_tokens/` for the SQLite image-token cache
- `squash/apertus-vllm-lmms-eval-13.0-prod.sqsh` through the TOML environment file

## Fan-Out Helpers

Run eval-only jobs for every model and task in the helper defaults:

```bash
bash scripts/eval_using_local_sqlite_copy.sh scripts/models.txt
```

Run eval-only jobs in read-only SQLite mode (no node-local copy):

```bash
bash scripts/eval_using_read_only_sqlite.sh scripts/models.txt
```

Fill or extend the shared SQLite cache first, if needed:

```bash
bash scripts/fill_shared_sqlite_cache.sh scripts/models.txt
```

Both helpers derive paths relative to this folder, while the Slurm directives inside the submitted `.slurm` files remain hardcoded to this production location.

## Common Overrides

Use environment variables with the helper scripts:

```bash
TASKS_CSV=realworldqa,seedbench \
BATCH_SIZE=512 \
NUM_PROCESSES=4 \
bash scripts/eval_using_local_sqlite_copy.sh scripts/models.txt
```

Read-only mode example with helper overrides:

```bash
TASKS_CSV=realworldqa,seedbench \
BATCH_SIZE=512 \
NUM_PROCESSES=4 \
bash scripts/eval_using_read_only_sqlite.sh scripts/models.txt
```

Use CLI flags with direct `sbatch`:

```bash
MODEL_PATH="$(awk 'NF && $1 !~ /^#/ {print; exit}' scripts/models.txt)"
sbatch scripts/run_lmms_eval_vllm_eval_local_copy.slurm \
  --model-path "${MODEL_PATH}" \
  --tasks realworldqa,seedbench \
  --batch-size 512 \
  --num-processes 4 \
  --enable-wandb false
```

Direct `sbatch` launch in read-only mode (no node-local SQLite copy):

```bash
MODEL_PATH="$(awk 'NF && $1 !~ /^#/ {print; exit}' scripts/models.txt)"
sbatch scripts/run_lmms_eval_vllm_eval_local_copy.slurm \
  --model-path "${MODEL_PATH}" \
  --tasks realworldqa,seedbench \
  --image-token-cache-local-copy 0 \
  --image-token-cache-readonly 1 \
  --image-token-cache-write-misses 0 \
  --enable-wandb false
```

## W&B

W&B is off by default in these production scripts so the basic commands work without a secret. To enable it, provide the key through the environment or `--wandb-api-key`:

```bash
export WANDB_API_KEY="..."
MODEL_PATH="$(awk 'NF && $1 !~ /^#/ {print; exit}' scripts/models.txt)"
sbatch scripts/run_lmms_eval_vllm_prod.slurm \
  --model-path "${MODEL_PATH}" \
  --tasks realworldqa \
  --enable-wandb true \
  --wandb-log-samples true
```

## Push Existing Results To W&B

Install `wandb` in your local Python environment, then run from this folder:

```bash
uv run python scripts/push_results_to_wandb.py \
  --results-root results \
  --latest-only \
  --dry-run
```

Remove `--dry-run` when the resolved `wandb_args` look correct. The push script uses launcher-compatible deterministic run IDs: `model_<sha1_12>`.

## Notes Before Launching

- `scripts/models.txt` paths must exist on compute nodes.
- `TOKENIZER_PATH` defaults to `/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct`.
- `IMAGE_TOKEN_CACHE_LOCAL_BASE` defaults to `/tmp/$USER/apertus_image_token_cache`; this is only used when local-copy mode is enabled.
- Slurm account and reservation are hardcoded as `infra01` and `SD-69241-apertus-1-5-0`.
- The container expects `/workspace/vllm`, `/workspace/lmms-eval`, and `/workspace/Emu3.5` as built into the sqsh image.

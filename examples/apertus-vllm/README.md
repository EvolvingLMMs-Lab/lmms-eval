# lmms-eval launch guide (Apertus + vLLM image-token cache)

This folder contains the launch scripts for running `lmms-eval` with Apertus and the SQLite-backed image tokenization cache.

All files referenced in this README are under:
`/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval`

Top-level locations you will use most often:

- `scripts/` for launchers and model list
- `cache/` for shared caches (including image-token cache)
- `squash/` for squash-related artifacts
- `logs/` for Slurm/job logs
- `results/` for evaluation outputs
- `responses/` for response artifacts

## Important note about the Docker image

`Dockerfile.vllm-lmms-eval-prod-cu130` contains the combined changes from `swiss-ai/vllm` + `lmms-eval` branch integration, including the image tokenization cache changes.

## Important files

- `Dockerfile.vllm-lmms-eval-prod-cu130`
  - Runtime image definition with the integrated vLLM/lmms-eval cache behavior.
- `apertus-vllm-lmms-eval-prod.toml`
  - Slurm environment file used by the `.slurm` launchers (`#SBATCH --environment=...`).
- `scripts/models.txt`
  - One model path per line. Lines starting with `#` are ignored.
- `scripts/fill_shared_sqlite_cache.sh`
  - Submission helper for cache-fill jobs (model x task fan-out).
- `scripts/eval_using_local_sqlite_copy.sh`
  - Submission helper for eval-only jobs using existing cache (model x task fan-out).
- `scripts/run_lmms_eval_vllm_fill_shared_cache.slurm`
  - Cache-fill launcher (full runner) with defaults pinned to this repo (`cache/`, `logs/`, `results/`).
- `scripts/run_lmms_eval_vllm_eval_local_copy.slurm`
  - Eval launcher (full runner) for local-copy + readonly cache mode, with defaults pinned to this repo.
- `scripts/run_lmms_eval_vllm_prod.slurm`
  - Main launcher with full argument parsing, env setup, preflight checks, and `accelerate launch`.
- `cache/`
  - Shared image-token SQLite cache location (default for both fill and eval helpers).
- `logs/`
  - Job logs.
- `results/`
  - Evaluation outputs.

## Quick start

From repo root:

```bash
cd /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval
```

### 1) Eval-only path (cache already filled)

This launches one Slurm job per `(model, task)` from `models.txt` and reuses the existing cache:

```bash
bash scripts/eval_using_local_sqlite_copy.sh scripts/models.txt
```

### 2) Cache-fill path

If cache is not populated for your models/tasks yet, run fill first:

```bash
bash scripts/fill_shared_sqlite_cache.sh scripts/models.txt
```

## Environment variables you are most likely to tune

### In `scripts/eval_using_local_sqlite_copy.sh`

- `SLURM_SCRIPT` (default: `scripts/run_lmms_eval_vllm_eval_local_copy.slurm`)
- `TOKENIZER_PATH`
- `TASKS_CSV`
- `IMAGE_TOKEN_CACHE_DIR` (default: `./cache`)
- `IMAGE_TOKEN_CACHE_LOCAL_BASE` (default: `/tmp/$USER/apertus_image_token_cache`)
- `LOG_DIR` (default: `./logs`)
- `SBATCH_OUTPUT`, `SBATCH_ERROR`
- `BATCH_SIZE`, `NUM_PROCESSES`

### In `scripts/fill_shared_sqlite_cache.sh`

- `SLURM_SCRIPT` (default: `scripts/run_lmms_eval_vllm_fill_shared_cache.slurm`)
- `TOKENIZER_PATH`
- `TASKS_CSV`
- `IMAGE_TOKEN_CACHE_DIR` (default: `./cache`)
- `LOG_DIR` (default: `./logs`)
- `SBATCH_OUTPUT`, `SBATCH_ERROR`
- `BATCH_SIZE`, `NUM_PROCESSES`

### In `scripts/run_lmms_eval_vllm_prod.slurm` (core launcher)

CLI options to know:

- `--model-path`, `--tokenizer-path`, `--tasks`
- `--output-path`, `--log-dir`
- `--num-processes`, `--batch-size`, `--main-process-port`
- `--gpu-memory-utilization`, `--max-model-len`, `--max-num-batched-tokens`
- `--gen-kwargs`, `--limit`, `--seed`
- `--enable-image-token-cache`, `--image-token-cache-dir`
- `--image-token-cache-collision-guard`
- `--image-token-memory-cache-size`
- `--image-token-sqlite-busy-timeout-ms`, `--image-token-sqlite-mmap-size`
- `--image-token-cache-debug`
- `--prefetch-emu35-vision-tokenizer`

Key inherited env variables used by this launcher:

- `HF_HOME`, `NLTK_DATA`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `LMMS_EVAL_MODELS_CACHE`
- `VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR`
- `VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD`
- `VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE`
- `VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS`
- `VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE`
- `VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG`

### Path behavior in the mode-specific `.slurm` scripts

- `scripts/run_lmms_eval_vllm_eval_local_copy.slurm` and `scripts/run_lmms_eval_vllm_fill_shared_cache.slurm` now default to this repo only:
  - `LOG_DIR=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/logs`
  - `IMAGE_TOKEN_CACHE_DIR=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/cache`
  - `HF_HOME`, `NLTK_DATA`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `LMMS_EVAL_MODELS_CACHE` under `.../lmms-eval/cache`
  - `OUTPUT_PATH=/capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/results`
- Their `#SBATCH --output` and `#SBATCH --error` are also pinned to `.../lmms-eval/logs`.

## Exact things to worry about before launching

1. `models.txt` paths must exist and be readable on compute nodes.
2. `cache/` must be writable for fill jobs and readable for eval jobs.
3. `IMAGE_TOKEN_CACHE_LOCAL_BASE` (eval-only mode) must have enough local disk space.
4. Slurm account/reservation are hardcoded in `.slurm` scripts:
   - `--account=infra01`
   - `--reservation=SD-69241-apertus-1-5-0`
   Update these if your allocation changes.
5. `#SBATCH --environment=.../apertus-vllm-lmms-eval-prod.toml` must point to a valid file.
6. `NUM_PROCESSES` cannot exceed visible GPUs; launcher will fail fast if it does.
7. `MAX_MODEL_LEN` must be `> 8192` (enforced by script).
8. The launcher expects repo layout under `/workspace` at runtime by default:
   - `VLLM_REPO=/workspace/vllm`
   - `LMMS_EVAL_REPO=/workspace/lmms-eval`
   If your container layout differs, export `VLLM_REPO` and `LMMS_EVAL_REPO` explicitly.
9. Emu3.5 preflight import is enforced; ensure `VLLM_APERTUS_EMU35_CODEBASE` is valid (default `/workspace/Emu3.5`).
10. Eval-only script pins readonly/no-write cache flags. If cache is missing entries for your task/model, you may lose cache benefits.

## Typical workflow

1. Edit `scripts/models.txt` with your model paths.
2. Run cache-fill once for new tasks/models.
3. Run eval-only for repeated evaluations using the warmed cache.
4. Collect outputs from `results/` and logs from `logs/`.

## Legacy absolute command equivalents

If you prefer absolute paths, the same commands are:

```bash
bash /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/scripts/eval_using_local_sqlite_copy.sh \
  /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/scripts/models.txt

bash /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/scripts/fill_shared_sqlite_cache.sh \
  /capstor/store/cscs/swissai/infra01/multimodal-eval/lmms-eval/scripts/models.txt
```

If your checkout still lives at the old location, these equivalent legacy commands are:

```bash
bash /iopsstor/scratch/cscs/anunay/swissai/apertus_integration/storage/eval_scripts/eval_using_local_sqlite_copy.sh \
  /iopsstor/scratch/cscs/anunay/swissai/apertus_integration/storage/eval_scripts/models.txt

bash /iopsstor/scratch/cscs/anunay/swissai/apertus_integration/storage/eval_scripts/fill_shared_sqlite_cache.sh \
  /iopsstor/scratch/cscs/anunay/swissai/apertus_integration/storage/eval_scripts/models.txt
```

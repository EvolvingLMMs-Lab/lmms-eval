# Issue #1563: reproduction and validation

Validated on 2026-10-05 against base
`e49214b08e77834c19ba0a53ee093979c6ffbe2c`.
The issue's private `${MODEL_PATH}` was not supplied. The reproduction uses
public `Qwen/Qwen3-VL-2B-Instruct` and the first two real MMLU abstract-algebra
questions through the same `qwen3_vl` / `multiple_choice` path.

## Environment and setup

Linux x86_64, Python 3.12.13, uv 0.11.33, torch 2.13.0 / CUDA 13.0,
transformers 5.17.0, datasets 5.0.1, accelerate 1.15.0,
huggingface-hub 1.32.0, tokenizers 0.23.2, qwen-vl-utils 0.0.14,
pytest 9.1.1, pre-commit 4.6.2; repository Ruff hooks pinned to 0.16.4.
One NVIDIA H800, 81,559 MiB, driver 595.71.05; bfloat16, eager attention,
batch size 1, `device_map=auto`.

```bash
uv sync --locked --inexact
```

This completed without changing dependencies or the lockfile. `--inexact`
preserves unrelated optional packages in this existing environment; a fresh
checkout can use `uv sync --locked`.

Pinned inputs:

- Model revision: `89644892e4d85e24eaac8bacfd4f463576704203`.
- `model.safetensors`: 4,255,140,312 bytes; SHA256
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
- Dataset: `hails/mmlu_no_train`, revision
  `b2e1ec9aa795adafe68e8e983248dbd4b52a1c60`, config `abstract_algebra`.
- Test split: 100 documents; selected indices 0 and 1, N=2. Five-shot context
  uses all five documents in the `dev` split and the task's `first_n` sampler.
- This dataset contains text questions and choices; no image/video inputs.

Prewarm the pinned dataset and checkpoint before the offline CLI commands:

```bash
./.venv/bin/python - <<'PYSETUP'
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lmms_eval.evaluator import _enable_reentrant_filelocks
_enable_reentrant_filelocks()
load_dataset('hails/mmlu_no_train', 'abstract_algebra',
             revision='b2e1ec9aa795adafe68e8e983248dbd4b52a1c60')
print(snapshot_download('Qwen/Qwen3-VL-2B-Instruct',
      revision='89644892e4d85e24eaac8bacfd4f463576704203',
      allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors']))
PYSETUP
```

The local snapshot path in the commands below is the original workspace path.
Replace it with the path returned by `snapshot_download` in another workspace;
replace the absolute interpreter path with that workspace's interpreter.
The final runs used cached pinned data and weights with Hub/network access
disabled. `datasets` logged its existing warning that `trust_remote_code` is
unsupported, then loaded the real cached dataset successfully.

Initial setup attempts encountered a slow native download and HTTP 429 from
parallel public resolver requests. The native attempt was interrupted; remaining
byte ranges were fetched from the Hub metadata's CDN location and assembled
only after verifying the complete SHA256 above. No model or dataset content
was modified. The exact download helper source is included at the end of this
report; normal `snapshot_download` is sufficient when downloads are available.

## Before: real CLI reproduces the reported assertion

```bash
git worktree add --detach /tmp/lmms-issue1563-base \
  e49214b08e77834c19ba0a53ee093979c6ffbe2c
cd /tmp/lmms-issue1563-base
env PYTHONPATH=/tmp/lmms-issue1563-base WANDB_MODE=disabled \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/umm/users/pufanyi/projects/lmms-eval/.venv/bin/python -m lmms_eval eval \
  --model qwen3_vl \
  --model_args pretrained=/mnt/umm/users/pufanyi/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203,max_pixels=3211264,attn_implementation=eager,interleave_visuals=False,device_map=auto \
  --gen_kwargs max_new_tokens=64,temperature=1.0,top_p=1.0,top_k=40,repetition_penalty=1.0 \
  --tasks mmlu_abstract_algebra --batch_size 1 --limit 2 --num_fewshot 5 \
  --output_path /tmp/issue1563-before --log_samples --seed 42 --verbosity DEBUG
```

Observed exit code 1, after loading the real model and dataset:

```text
Building contexts for mmlu_abstract_algebra on rank 0...
AssertionError: Currently messages is used for generation only
```

[before.log](before.log) contains the traceback. Removing the task restriction
alone still reaches the backend's unimplemented `loglikelihood`. An initial
patch that added request construction and Qwen3 likelihood scoring completed
all eight real choice scores but exposed a third bug: the evaluator's
unconditional generation-output normalizer converted numeric pairs to strings.
The choice scorer raised `ValueError: too many values to unpack (expected 2)`,
wrapped by `tenacity.RetryError`. This failed attempt is preserved in
[after-initial.log](after-initial.log).

## After: real CLI completes scoring and metrics

From the patched project checkout:

```bash
env WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./.venv/bin/python -m lmms_eval eval \
  --model qwen3_vl \
  --model_args pretrained=/mnt/umm/users/pufanyi/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203,max_pixels=3211264,attn_implementation=eager,interleave_visuals=False,device_map=auto \
  --gen_kwargs max_new_tokens=64,temperature=1.0,top_p=1.0,top_k=40,repetition_penalty=1.0 \
  --tasks mmlu_abstract_algebra --batch_size 1 --limit 2 --num_fewshot 5 \
  --output_path /tmp/issue1563-after --log_samples --seed 42 --verbosity DEBUG
```

For the simple adapter, the exact same command was repeated with
`--force_simple` after `--model qwen3_vl` and
`--output_path /tmp/issue1563-simple`.

Both commands exited 0, scored eight nonempty continuations, saved two samples,
and emitted `acc,none=0.0`. The result metadata identifies
`lmms_eval.models.chat.qwen3_vl.Qwen3_VL (chat)` and
`lmms_eval.models.simple.qwen3_vl.Qwen3_VL (simple)` respectively. All eight
floating-point scores and the metrics match exactly between adapters.
Generation kwargs match the issue's settings with an explicit
`max_new_tokens=64`; generation settings do not affect likelihood scoring.
The model is evaluated with teacher-forced continuation probabilities, rather
than free-form answer generation.

| Test document | NLL A | NLL B | NLL C | NLL D | Selected by numeric argmin | Gold |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 23.234174728393555 | 25.734174728393555 | 25.359174728393555 | 21.046674728393555 | D | B |
| 1 | 21.91903305053711 | 28.48153305053711 | 26.82528305053711 | 26.41903305053711 | A | C |

All `is_greedy` flags are false and token-count entries are null: likelihood
scoring does not claim generation token usage. The existing sample logger
serializes score/boolean scalars as strings in JSONL; the report converts them
back to floats/bools for the table and derives labels using numeric argmin.
The in-memory evaluator preserves the numeric pairs, as the regression tests
verify. Sample serialization is unchanged by this PR.

Evidence: [chat-after.log](chat-after.log), [simple-after.log](simple-after.log),
and [validation.json](validation.json), including full five-shot inputs,
all actual scores, metrics, pinned revisions, environment and runtime source
SHA256 values. Log copies remove ANSI colors and intermediate weight-loading
progress updates; final progress and diagnostic lines are retained.
This is a smoke test for a working evaluation path, not evidence of model
accuracy improvement or a full MMLU run. The private checkpoint, image/video
likelihood, custom-message likelihood, Qwen3.5 and MoE models were not evaluated.

## Regression and static checks

New tensor-based model tests independently compute expected causal summed NLL
from deterministic next-token logits. They cover A/B/C/D rankings, multi-token
continuations, prompt-loss exclusion, causal shifting, joint BPE boundary
changes, greedy flags across all continuation tokens, system/thinking template
settings, full few-shot context, callback targets, two-argument requests,
empty requests/targets, and explicit media rejection, for simple and chat
adapters. Request tests compare the production layouts and cold/warm request
cache behavior. Evaluator tests run actual choice aggregation with both live
tuples and JSON lists from a real SQLite response cache, then verify that
plain/typed generation output and token-count normalization still work.

Against the base implementation, the final model/request test files reproduced
`25 failed, 25 passed in 48.11s` (exit 1):

```bash
cd /tmp/lmms-issue1563-base
env PYTHONPATH=/tmp/lmms-issue1563-base CUDA_VISIBLE_DEVICES= \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  OPENAI_API_KEY=lmms-eval-ci-placeholder \
  /mnt/umm/users/pufanyi/projects/lmms-eval/.venv/bin/python -m pytest \
  --import-mode=importlib -o pythonpath=/tmp/lmms-issue1563-base -q \
  /mnt/umm/users/pufanyi/projects/lmms-eval/test/models/test_qwen3_vl_likelihood.py \
  /mnt/umm/users/pufanyi/projects/lmms-eval/test/eval/test_request_construction_contract.py
```

Before fixing evaluator response handling, running the evaluator regression
file on the initial patch produced `2 failed, 8 passed in 66.95s`, reproducing
both uncached and cached numeric-pair failures. An earlier test-only fixture
omitted `features`, producing two `AttributeError` failures; it was corrected
before the final before/after runs. These checks use no API or downloaded data.

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OPENAI_API_KEY=lmms-eval-ci-placeholder \
  ./.venv/bin/python -m pytest -q \
  test/models/test_qwen3_vl_likelihood.py test/models/test_qwen3_vl_simple.py \
  test/eval/test_request_construction_contract.py test/eval/test_protocol.py \
  test/eval/test_task_pipeline.py test/eval/test_evaluator.py \
  test/eval/test_evaluator_sample_limits.py test/cache
# 248 passed, 17 subtests passed in 57.95s; exit 0
uv run pre-commit run --all-files
# ruff check Passed; ruff format Passed; exit 0
git diff --check
# exit 0
```

The CI hermetic CPU contracts include the new model tests and both modified
request/evaluator test files. Initial pre-commit formatting adjusted one new
test file; the final full invocation passed. No runtime sources changed after
the successful CLI runs; their exact SHA256 values are in the JSON report.

## Download fallback helper source

The CDN location was obtained without displaying its temporary signed URL:

```bash
./.venv/bin/python - <<'PYURL'
from pathlib import Path
from huggingface_hub import get_hf_file_metadata
url = 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/model.safetensors'
metadata = get_hf_file_metadata(url)
path = Path('/tmp/issue1563-download-location')
path.touch(mode=0o600, exist_ok=True)
path.write_text(metadata.location)
PYURL
```

The following is the full temporary helper used after prewarming checkpoint
metadata. Run it with the project interpreter if normal downloads are blocked.
It validates HTTP byte ranges and the final digest before installing weights
in the standard Hub cache. The temporary signed URL is not included in evidence.

```python
"""Download the public test checkpoint by verified parallel byte ranges."""
import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
SIZE = 4255140312
DIGEST = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
URL = f"https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/{REVISION}/model.safetensors"
DOWNLOAD_URL = Path("/tmp/issue1563-download-location").read_text()
ROOT = Path("/tmp/issue1563-checkpoint-parts")
ROOT.mkdir(exist_ok=True)
CHUNK_SIZE = 64 * 1024**2

def fetch(index):
    start = index * CHUNK_SIZE
    end = min(SIZE, start + CHUNK_SIZE) - 1
    output = ROOT / f"{index:03d}.part"
    if output.exists() and output.stat().st_size == end - start + 1:
        return output
    for attempt in range(3):
        try:
            with requests.get(DOWNLOAD_URL, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{end}/{SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Wrong range for part {index}: {response.status_code}, {response.headers.get('Content-Range')}")
                with output.open("wb") as file:
                    for block in response.iter_content(1024**2):
                        file.write(block)
            if output.stat().st_size != end - start + 1:
                raise RuntimeError(f"Incomplete part {index}")
            return output
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

started = time.monotonic()
parts = list(range((SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE))
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(fetch, index) for index in parts]
    for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
        future.result()
        print(f"Completed {completed}/{len(parts)} ranges in {time.monotonic() - started:.1f}s", flush=True)

assembled = ROOT / "model.safetensors"
digest = hashlib.sha256()
with assembled.open("wb") as file:
    for index in parts:
        with (ROOT / f"{index:03d}.part").open("rb") as source:
            while block := source.read(4 * 1024**2):
                digest.update(block)
                file.write(block)
assert assembled.stat().st_size == SIZE
assert digest.hexdigest() == DIGEST
cache = Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-VL-2B-Instruct"
target = cache / "blobs" / DIGEST
if not target.exists():
    import shutil
    temporary = target.with_name(DIGEST + ".issue1563-verified")
    shutil.copyfile(assembled, temporary)
    os.replace(temporary, target)
pointer = cache / "snapshots" / REVISION / "model.safetensors"
if not pointer.exists():
    pointer.symlink_to(Path("../../blobs") / DIGEST)
print("Verified SHA256:", DIGEST, flush=True)
print("Cached checkpoint:", hf_hub_download("Qwen/Qwen3-VL-2B-Instruct", "model.safetensors", revision=REVISION, local_files_only=True), flush=True)
```

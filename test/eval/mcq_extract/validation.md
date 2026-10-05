# Reproduction and validation

Validated on 2026-10-05 against merged #1554 at
`bc27dfacaa2d23f3203544e1c0cc3141964167e3`. The tests were specified from
the contract before implementation alignment; test and fix agents communicated
only with the parent. The fix agent could read the independently authored JSON.

## Environment and setup

Linux, Python 3.12.13, uv 0.11.33, NVIDIA H800 (81,559 MiB), torch 2.13.0
with CUDA 13.0, torchvision 0.28.0, transformers 5.17.0, datasets 5.0.1,
accelerate 1.15.0, qwen-vl-utils 0.0.14, pytest 9.1.1, pre-commit 4.6.2,
huggingface-hub 1.32.0. Ruff's pre-commit hooks use the repository-pinned
0.16.4; the project environment also has ruff 0.16.8.

```bash
uv sync --locked --inexact
```

The `--inexact` flag preserves unrelated optional packages in the existing
workspace environment. Sync completed successfully and rebuilt the editable
project without changing the lockfile. A fresh environment can use the
repository's `uv sync --locked` setup instead.

## Corpus reproduction

The committed helper contains the complete reproduction source. It reads the
same `cases.json` for the historical module and the final working tree.

```bash
./.venv/bin/python test/eval/mcq_extract/reproduce.py \
  --revision bc27dfacaa2d23f3203544e1c0cc3141964167e3 \
  --output /tmp/mcq-before.json
# exit 1: total=219, passed=122, failed=97
./.venv/bin/python test/eval/mcq_extract/reproduce.py \
  --output /tmp/mcq-after.json
# exit 0: total=219, passed=219, failed=0
```

All 26 unique legacy inputs match their desired expectations on the base.
All six regression seeds reproduce their recorded `base_expected` outputs.
Per-case actual before/after outputs are committed in [validation.json](validation.json).
The 97 historical mismatches include new format support and intentional policy
changes, not just regressions introduced by #1554.

## CPU validation

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  OPENAI_API_KEY=lmms-eval-ci-placeholder \
  ./.venv/bin/python -m pytest -q -m 'not gpu and not api' \
  test/eval/test_mcq_extract.py \
  test/eval/test_mcq_extract_consumers.py \
  test/eval/test_cgbench.py test/eval/test_kaleidoscope.py \
  test/eval/test_longvideo_reason.py test/eval/test_mmr_v.py \
  test/eval/test_vrbench.py test/eval/test_relax_exact_match_mcq.py \
  test/eval/test_task_pipeline.py
```

Observed output: `2009 passed in 117.27s`.

The MCQ suite accounts for 1,787 checks: 219 corpus inputs through each of
the utility and verifier wrapper, one fixture integrity check, 1,066
letter/format substitutions, 114 additional invariants, and 168 actual
consumer/API integration checks. It covers multiple choice counts and
noncontiguous alphabets. The remaining checks exercise existing task consumers
and benchmark-specific official scorers.

An initial invocation omitted the CI placeholder key and produced
`2 failed, 2007 passed`: existing MMVet task imports constructed an OpenAI client
and raised `openai.OpenAIError: Missing credentials`. Repeating the entire
command with the workflow's existing placeholder configuration passed. These
CPU checks do not call an API or count as model E2E evidence.

```bash
uv run pre-commit run --all-files
# ruff check: Passed; ruff format: Passed
git diff --check
# exit 0
```

## Real model E2E

- E2E status: PASS
- Model/backend: Qwen/Qwen2.5-VL-3B-Instruct, `qwen2_5_vl` backend;
  model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Dataset split and sample size: Lin-Chen/MMStar, val, indices 0–3, N=4;
  dataset revision `bc98d668301da7b14f648724866e57302778ab27`.
- Hardware: NVIDIA H800; bfloat16, SDPA, one GPU, batch size 1.
- Generation: greedy (`do_sample=False`), max_new_tokens=64, seed=42.

Run the historical baseline with the same environment and arguments:

```bash
git worktree add --detach /tmp/lmms-mcq-e2e-base \
  bc27dfacaa2d23f3203544e1c0cc3141964167e3
cd /tmp/lmms-mcq-e2e-base
env PYTHONPATH=/tmp/lmms-mcq-e2e-base WANDB_MODE=disabled HF_HUB_DISABLE_XET=1 \
  /mnt/umm/users/pufanyi/projects/lmms-eval/.venv/bin/python -m lmms_eval eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,attn_implementation=sdpa,device_map=cuda:0 \
  --tasks mmstar --batch_size 1 --limit 4 \
  --gen_kwargs max_new_tokens=64,do_sample=False \
  --output_path /tmp/mcq-e2e-before --log_samples --seed 42
```

The absolute interpreter path is the original validation workspace; substitute
the interpreter in your own project environment when reproducing. In the
patched checkout, run:

```bash
env WANDB_MODE=disabled HF_HUB_DISABLE_XET=1 \
  ./.venv/bin/python -m lmms_eval eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,attn_implementation=sdpa,device_map=cuda:0 \
  --tasks mmstar --batch_size 1 --limit 4 \
  --gen_kwargs max_new_tokens=64,do_sample=False \
  --output_path /tmp/mcq-e2e-after --log_samples --seed 42
```

Both CLI runs exited 0, generated four nonempty predictions from real images,
scored all four samples, and emitted `mmstar average=0.0`. Predictions in both
runs were D, C, B, B; targets were A, B, D, C. This is a path/integration smoke
test, not evidence of a model accuracy improvement. These generations are bare
uppercase answers; the corpus independently verifies all changed formats.

Observed patched log excerpt:

```text
Selected Tasks: ['mmstar']
Model Responding: 100%|██████████| 4/4
Postprocessed 4 docs for mmstar/none with 4 worker(s) in 0.02s
|mmstar|none|0|average|↑|0|±|N/A|0|
```

The per-sample record for document 0 contains `input_media: ["images/0.jpg"]`,
`filtered_resps: "D"`, target `"A"`, and score `0.0`. All four observed
prediction/media/target/score records and artifact paths are in
[validation.json](validation.json). Original logs remain at
`/tmp/mcq-e2e-before.log` and `/tmp/mcq-e2e-after.log`; sample records and result
JSONs are under `/tmp/mcq-e2e-before` and `/tmp/mcq-e2e-after`.

E2E ran on the final working tree before committing. Its extractor SHA-256,
recorded in the evidence JSON, is
`c38447cf7927bc9e9498692a544d98bf50a3cdd4c7d09adca5793eb748a9739d`.
No runtime edits followed that run. Plain `mmstar` calls the shared utility;
the separate `mmstar_reasoning` parser was not used for this evidence.

## Existing libraries

The optional Math-Verify comparison, its exact command, and the decision about
using existing library helpers are documented in the [contract README](README.md#existing-library-comparison).

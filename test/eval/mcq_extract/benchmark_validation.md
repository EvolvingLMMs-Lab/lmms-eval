# Benchmark attack fixture validation

Base revision: `00a12d40aff8ded19c37fbcf8e7d273034f63f81` (merged #1564).
Only test data, test code, diagnostics, and documentation change here. The
shared extractor already satisfies the new expectations; this extends its
regression coverage without modifying any benchmark's native scorer.

## Environment and setup

Existing uv-managed `.venv`, prepared with `uv sync --locked --inexact` to
preserve unrelated optional packages. A fresh checkout can use
`uv sync --locked`. No dependency or lockfile change is required.

| Component | Observed version |
| --- | --- |
| Python | 3.12.13 |
| uv | 0.11.33 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| pytest | 9.1.1 |
| pre-commit | 4.6.2 |
| Ruff hooks | 0.16.4 (repository-pinned) |

Hardware: CPU-only parsing/tests on the existing Linux workspace. No dataset
download, model inference, or judge API was used for these observations.

## Inputs and before/after coverage

The existing corpus contains 219 fixtures at the base and no `hack_*`
categories. The expanded corpus contains 271 fixtures, including 52 attacks:
12 `hack_mmmu` and eight each of `hack_mmmu_pro`, `hack_videommmu`,
`hack_mmbench`, `hack_seedbench`, and `hack_ai2d`.

Exact coverage comparison command:

```bash
./.venv/bin/python - <<'PY'
import json
import subprocess
from collections import Counter
base = "00a12d40aff8ded19c37fbcf8e7d273034f63f81"
old = json.loads(subprocess.check_output(
    ["git", "show", f"{base}:test/eval/mcq_extract/cases.json"], text=True
))["cases"]
new = json.load(open("test/eval/mcq_extract/cases.json"))["cases"]
for label, cases in (("base", old), ("working tree", new)):
    attacks = Counter(c["category"] for c in cases if c["category"].startswith("hack_"))
    print(label, "cases=", len(cases), "hack_categories=", dict(attacks),
          "hack_total=", sum(attacks.values()))
PY
```

Observed output: base `cases=219, hack_total=0`; expanded `cases=271,
hack_total=52`. Expectations were written from the existing shared extraction
contract before running the new fixtures. None of the native heuristics were
used to generate the desired answer.

## Native parser observations

```bash
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled \
  ./.venv/bin/python test/eval/mcq_extract/probe_benchmarks.py \
  --output /tmp/mcq-hack-benchmark-observations.json
```

Observed summary: `total=52`, `shared_passed=52`,
`native_disagreement_cases=50`. The complete raw predictions, seeds, runtime
versions, source revision, and corpus hash are in
[benchmark_validation.json](benchmark_validation.json). To regenerate that
artifact in place, use its path as `--output`.

Each fixture supplies exact synthetic option text in `benchmark_context`.
The probe calls real local parser functions, MMBench's real prefetch path,
SEEDBench's result processor, or AI2D's configured filter. It does not mock
their results. It preserves and restores Python's global random state, and
replays randomized MMMU-family paths at seeds 0, 1, 7, and 42.

Representative observed differences:

| Category | Exact response | Native output | Shared expected/observed |
| --- | --- | --- | --- |
| `hack_mmmu` | `The diagram labels an earlier candidate (A). The answer is B.` | A | B |
| `hack_mmmu` | `I cannot determine which option matches the image.` | D/B/C/A at seeds 0/1/7/42 | empty |
| `hack_mmmu_pro` | `Answer: JUNK` | J | empty |
| `hack_videommmu` | `The notice reads DNA. It does not select an option.` | A | empty |
| `hack_mmbench` | `B or C.` | B | empty |
| `hack_seedbench` | `ABCD` | A | empty |
| `hack_ai2d` | `A. Earlier draft.\nThe answer is D.` | A | D |

These counts measure disagreement with this project's chosen extraction
contract, not benchmark accuracy or the authors' intended scoring policies.
For randomized paths, a case counts as a disagreement if at least one of
the recorded seeds differs. MMBench API judging is not exercised. AI2D's raw
filter output is recorded before metric normalization. Native lowercase
single-choice letters are canonicalized for comparison; raw non-label outputs
remain visible in the JSON and summarize to no selection.

Two fixtures return an unoffered E in their native paths. Those are boundary
controls, not examples that can score correctly against an A-D target, and
are excluded from the 50 disagreement count. Native observations are not
asserted as golden outputs, so fixing those scorers later will not require
preserving these behaviors in tests.

## Shared contract tests

```bash
./.venv/bin/python test/eval/mcq_extract/reproduce.py \
  --output /tmp/mcq-hack-contract.json
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  OPENAI_API_KEY=lmms-eval-ci-placeholder \
  ./.venv/bin/python -m pytest -q \
  test/eval/test_mcq_extract.py test/eval/test_mcq_extract_consumers.py
uv run pre-commit run --all-files
git diff --check
git diff --cached --check
```

Observed corpus output: `total=271`, `passed=271`, `failed=0`, exit 0.
Observed pytest output: `1891 passed in 52.32s`, exit 0. Every new fixture
runs against both the shared utility and `MCQExtractor` through the existing
JSON-driven tests. Attack context is also checked for matching option keys
and category names. Existing consumer and generated invariant tests pass.
Both full-repository pre-commit hooks passed; whitespace checks passed.

E2E status: NOT APPLICABLE. This change only adds test data, diagnostics,
test assertions, and documentation. It changes no task configuration,
prompt, model, production parser, dependency, or evaluation behavior. The
historical real model validation for the unchanged extractor is retained in
`validation.md`; no new model inference is claimed here.

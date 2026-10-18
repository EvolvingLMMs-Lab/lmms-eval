# DIVE-Bench and GRT

[Project / leaderboard](https://www.zhanghaichao.xyz/DenseVideoUnderstand/) ·
[Public code and audit](https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/release/dive-bench-minimal) ·
[Earlier public paper](https://arxiv.org/abs/2509.14199)

This integration provides DIVE-Bench's objective task metrics and isolated GRT
(Gated Residual Tokenization) inference wrappers. It does not modify the stock
`llava_hf` or `qwen2_5_vl` models. GRT reuses eligible video patch projections;
it does not reduce the visual sequence length or prove an end-to-end FLOP or
latency reduction by itself.

## High-Motion target/reference hold — 14 September 2026

**All High-Motion quality rankings are withheld**, including previously
protocol-screened baselines. An independently confirmed check of four canonical
references found that their archived trajectories match a
`leftIndexFingerMetacarpal` projection, while the question requests the
right-hand palm/ring-base target. The affected extent across the 3,243 items and
the original label constructor remain under review; this does not establish
that every item is affected.

Successful runtime/smoke checks or reproduction of archived metrics cannot
resolve this target/reference mismatch or establish High-Motion quality or GRT
superiority. High-Motion commands below are retained for authorized implementation
checks, not approved quality comparisons. See the public
[High-Motion hold and review conditions](https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/blob/release/dive-bench-minimal/docs/HIGHMOTION_TARGET_HOLD.md).

This documentation-only notice changes no task code, data pins, prompts,
reference labels, sampling, metrics, or task semantics. Educational instructions
and all existing installation settings remain unchanged. Both framework
integrations remain **Draft PRs**:
[VLMEvalKit #1686](https://github.com/open-compass/VLMEvalKit/pull/1686) and
[lmms-eval #1521](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1521);
neither is merged or accepted.

## Tasks and protocols

| Task | Scenario in the revised manuscript | Examples | Compatibility alias |
| --- | --- | ---: | --- |
| `dive_bench_educational_high_fps` | Educational High-FPS Videos | 634 questions / 317 videos | `densevideo` |
| `dive_bench_high_motion_high_fps` | High-Motion High-FPS Videos | 3,243 clips | — |
| `dive_bench_high_motion_high_fps_preview1000` | High-Motion High-FPS Videos, published preview | first 1,000 clips | `densevideo_highmotion` |

The full and preview tasks must not be mixed on one leaderboard. The preview is
always the first 1,000 rows; the legacy `DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES`
environment variable cannot silently change its meaning. The shared `qid` values
in high-motion are reused across action directories: use harness row IDs and
preserve `video_path`, not `qid` alone, when joining results.

The public arXiv v2 manuscript describes an earlier educational-only benchmark.
The two scenario names above follow the revised manuscript distributed with the
public code; they do not imply that its additional high-motion content is already
present in arXiv v2. Counts, judge protocol and sampling also differ between old
paper tables and the current released annotations. Do not claim paper-table
reproduction solely from a task name.

## Data access and installation

The educational source repository is gated. On 2026-09-14, an owner-authenticated
audit verified that the pinned high-motion source repository is **private** and
downloaded its annotations. **A public end-to-end download/inference run is not
yet verified.** Successful owner access does not establish public access.
Dataset access must be granted by the owners; authentication alone may not suffice.
The integration does not redistribute videos or annotations or change access.

- [Educational data](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)
  is pinned to revision `5cc61a045c8e5e95d1d9c87e22ccd0f699575aea`. Only
  `LPM_videos.parquet` is loaded; `LPM_slides.parquet` is an identical duplicate
  and loading both would double the question count.
- [High-motion data](https://huggingface.co/datasets/haichaozhang/highmotion_densevideounderstand)
  uses `Egodex_traj.parquet`, pinned to owner-verified revision
  `d44407f607fdf020c59b816884f06ed6d453cf26`. The Hub file has SHA256
  `518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd`.
  The audited historical local serialization has SHA256
  `39f9da7aca9020d79f383953646a5893f09c6f8e5f60433560011280ee987b2d`.
  Both contain the same 3,243 distinct video paths and ordered task content.
  The loader also checks the ordered content hash of `(video_path, qid,
  question, answer, frame_count)` against
  `90ee915016105f6a709f391e8a03a6d0e99bc5c908f945cdf7b80d0cb289e789`.
  A changed or reordered same-size table therefore fails closed, while differences
  in Parquet writer metadata/compression alone do not invalidate task content.
- Educational source terms include [LPM's CC BY-NC-SA 4.0 data terms and original
  YouTube video terms](https://github.com/dondongwon/LPMDataset#license).
  [EgoDex data](https://github.com/apple/ml-egodex#license) has CC BY-NC-ND terms.
  This repository's code license does not supersede these asset terms.

On 2026-09-14, the owner configured the **private** canonical
[`haichaozhang/DIVE-Bench`](https://huggingface.co/datasets/haichaozhang/DIVE-Bench)
repository at revision `d80461fccf879d5efdeece0edce8608a72d64f10` with exactly two
annotation configurations: `educational_high_fps` (634 rows) and
`high_motion_high_fps` (3,243 rows). Named configurations were added to the old
source cards at revisions `c3ff65dfc37239ebee05bd190cfa5b5126f49146` (educational)
and `25cc1aeaef5209776625ce4e72a3ba425d4ae929` (high-motion), leaving non-card
objects and access settings unchanged. This integration keeps its original
immutable source-data pins. This is completed annotation configuration, not a
public-download, video-redistribution, or end-to-end GPU reproduction claim.

After authorized download, extract the video files manually under one root:

```text
/data/dive-bench/
  DenseVideo-LPM/videos/<video>.mp4
  egodex/<action>/<clip>.mp4
```

Set `DIVE_BENCH_DATA_ROOT=/data/dive-bench`. The educational archive's alternative
`videos/<video>.mp4` layout is accepted. For high-motion, action directories are
mandatory: a basename-only fallback could silently score the wrong clip. Missing
videos fail with an actionable error. Annotation loading does not automatically
download the 27 GB educational video archive.

Install the GRT extra into a fresh environment, separately from other models that
require different Transformers versions:

```bash
uv venv --python 3.10 .venv-grt
uv pip install --python .venv-grt/bin/python -e '.[grt]'
```

The verified model-internal patching targets Transformers 4.57.6 and fails closed
for other versions before loading weights. The optional extra also constrains
PyTorch 2.9, Torchvision 0.24, Accelerate 1.x, PyAV 15 and qwen-vl-utils 0.0.14.
The published educational profiles use the PyAV loader and do not require Decord.
No training or benchmark model weights are bundled.

## Run an educational GRT profile

Three model identifiers are registered lazily:

| Model ID | Use |
| --- | --- |
| `grt_llava_hf` | LLaVA-OneVision 0.5B Route31 and its matched controls |
| `grt_qwen2_5_vl` | Qwen2.5-VL 3B t03 and non-floor Qwen controls |
| `grt_qwen2_5_vl_floor` | Qwen2.5-VL 7B subtitle/OCR route-floor candidate |

These registered GRT adapters accept only `dive_bench_educational_high_fps` and
its `densevideo` alias. They reject every High-Motion alias, unknown task, and
batch containing any unsupported task before routing, decoding, or backend inference. This guard also
applies to direct native CLI calls, not just the profile helper; the strict worker
additionally checks task flags before importing its CUDA runtime. There is no
bypass flag. Raw High-Motion tasks, stock models, metrics, and historical evidence
remain available; they are not released GRT model profiles.

Use the bundled profiles rather than generic defaults. They include checkpoint
revisions, eight-frame budgets, 384-pixel resizing, thresholds, the Route31
subtitle-specific 31-token cap and the Qwen7 48-token cap. `qwen3` and `qwen7`
refer to **Qwen2.5-VL model sizes**, not the Qwen3 model family.

This prints a command without downloads, model loading or filesystem writes:

```bash
.venv-grt/bin/python -m lmms_eval.models.model_utils.grt.profile \
  --profile route31 --role candidate --output ./runs/route31-session-1
```

Inspect and run the printed command. It includes the strict worker and native
lmms-eval CLI flags. Set `CUDA_VISIBLE_DEVICES` to exactly one GPU and keep the
same physical GPU for **serial** `base`, `all`, and `candidate` runs. Repeat for
profiles `qwen3` and `qwen7`. Each output path must be new. Save each command's
stdout/stderr alongside its sample log: `[DENSE_METRICS]` records patch counts
and sampling/throughput telemetry. The worker records package and device details.
The bootstrap seed is 0; explicit evaluator seeds are `0,1234,1234,1234`, matching
the historical evaluator defaults. TF32 is disabled and deterministic algorithms
are required, so unsupported deterministic operations fail instead of silently
falling back.

The strict worker rejects distributed world-size/rank environment settings and
an already initialized multi-process group, even if each process sees one GPU.
It accepts the explicit flags printed by the profile helper, not `--config`
files that can override the checked environment. It forces the native CLI's
`DEBUG` exception-propagating mode: data/model/CUDA failures therefore exit
nonzero instead of being logged and swallowed by upstream's ordinary `INFO`
mode. Debug logs can include dataset text and predictions; keep them private
unless the source terms permit sharing them. These are launch/error-handling
guards, not changes to GRT kernels or generation parameters.

These commands generate predictions and native objective metrics. They do not
run an Open MOS judge or automatically qualify a candidate for the leaderboard.
Use the public code's separate audited judge and quality-gate workflow, which
compares against matched controls as well as archived public baselines.

## Evaluate high-motion

The high-motion metric target uses endpoint-inclusive uniform frame indices
`linspace(0, F-1, min(K, F)).astype(int)`, with `K=8` by default. Set
`DENSEVIDEO_HIGHMOTION_NUM_FRAMES` only together with the model's frame budget.
The caller must verify that its video decoder uses the same indices; a generic
video model that instead uses an FPS policy or excludes the final frame is not
protocol-equivalent. The task name's High-FPS scenario label does not mean this
eight-frame evaluation processes a video densely.

For raw-task diagnostics with an independently installed stock model (not a
released GRT profile, approved benchmark result, or historical reproduction):

```bash
export DIVE_BENCH_DATA_ROOT=/data/dive-bench
export DENSEVIDEO_HIGHMOTION_NUM_FRAMES=8
python -m lmms_eval \
  --model YOUR_STOCK_VIDEO_MODEL \
  --model_args YOUR_MODEL_SPECIFIC_ARGS \
  --tasks dive_bench_high_motion_high_fps_preview1000 \
  --batch_size 1 --log_samples --output_path ./runs/highmotion-diagnostic
```

Replace the placeholders using that stock adapter's own installation and argument
documentation. The task's eight-frame setting does not configure the stock
decoder or establish endpoint-inclusive sampler alignment. Verify its actual
frame indices separately before interpreting diagnostic metrics, and keep
predictions private. The target/reference-consistency hold remains in force.

The historical website high-motion GRT row used a different
`llava_ov_dense_video` wrapper and did not freeze a complete dirty-source snapshot
or model revision. This integration does not relabel that result as one produced
by the new wrapper IDs. A full 3,243-row evaluation is a different result from the
published 1,000-row preview.
The legacy wrapper also fails the aligned eight-frame, full-clip protocol: the
source audit found fewer than eight input frames on 787 of the first 1,000 clips
and a ten-second truncation affecting 148 clips. The historical row is therefore
not a fair matched-protocol comparison against eight-frame baselines and cannot
establish that GRT outperforms them. These counts audit the historical sampler,
not fresh GPU generations; the integration changes neither its scores nor the
GRT algorithms.

## Metrics and validation boundary

Educational CER/WER, multiset Token F1 and Exact Match normalize outer/collapsed
whitespace and case, retaining punctuation. CER/WER are exact per-example edit
distances divided by reference length (minimum denominator one); values can
exceed one. All examples are scored even when historical fast-metric environment
variables are set. Aggregation is the mean of per-question values.

High-motion Grid Accuracy, ADE, FDE and transition accuracy use canonical 3×3
cell centers. Missing predicted positions get no correct-label credit and a
distance penalty of `sqrt(2)`; extra positions do not alter spatial displacement
metrics. Token F1 compares canonical labels as a multiset and is order-insensitive.
Metrics are averaged per clip. No GPT/Open MOS metric is registered: a disabled
or failed judge is never emitted as a real zero MOS score.

CPU tests cover native task schemas/loading, all five task IDs, three model IDs,
path collision protection, metric boundary cases, GRT reuse/refresh behavior,
native all-keep equivalence and route-floor fallback. Function AST fingerprints
in `test/dive_bench/function_parity.json` preserve 142 pre-port function bodies,
including the GRT patch kernels and high-motion prompt/scoring functions.
Porting changes outside these frozen bodies are import relocation, additive
registration, explicit dependency guards and native task adapters. Four reviewed
lint-only edits are `type(x) == str` → `type(x) is str` in two legacy paths,
removing an unused local list in Qwen generation, and narrowing a legacy decoder
bare exception to `Exception`. No claims of a fresh GPU full-benchmark or MOS
rerun are made by these CPU checks.

The four model-kernel test modules are collected only in the pinned GRT runtime.
This does not require upstream's stock/newer Transformers test environment to
downgrade; native task, metric, schema and AST tests remain available there.

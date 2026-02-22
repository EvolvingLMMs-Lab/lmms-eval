# Changelog

## v0.6 (2026-02-16)

### Highlights

- **~7.5x API throughput improvement** via adaptive concurrency control, refill scheduling, prefix-aware queueing, and retry/backoff decoupling
- **Statistical analysis framework**: confidence intervals, clustered standard errors, paired comparison, power analysis, and model stability metrics
- **Evaluation as a Service**: HTTP eval server for async job submission decoupled from training loops
- **Model Registry V2**: manifest-driven unified model resolution with backward-compatible aliasing
- **50+ new evaluation tasks** and **10+ new model integrations**
- **Minimum Python version**: raised to 3.10

### Architecture & Performance

- **Adaptive concurrency control** for API-backed evaluation (`async_openai`, `openai`). The controller adjusts in-flight request count using three online signals: failure rate, rate-limit hit rate (429/throttling), and p95 latency against a target budget. Measured ~7.5x throughput gain over static single-concurrency baseline on `mme` with `LIMIT=100`. (#1080, #1082)
- **Refill-style scheduling**: completed requests immediately release slots for new work, eliminating the full-window barrier where the slowest request gates the entire batch.
- **Prefix-aware queueing**: reorder dispatch by prefix hash so same-prefix requests are sent close together, improving prefill-cache hit opportunities on providers that support prefix caching. (#1080)
- **Retry/backoff decoupling**: `retry_backoff_s` is explicitly separate from request timeout, so retries don't sleep for long timeouts and tie up worker slots.
- **Throughput metrics in results table**: final output now includes requests/sec and wall time for each task. (#1078)
- **`--offset` option**: skip the first N samples in a dataset, useful for resuming partial runs or debugging specific subsets. (#1042)

### Model Registry V2

- **Manifest-driven model resolution** (`ModelRegistryV2`): all model names resolve through a single path. Two dicts in `models/__init__.py` (`AVAILABLE_SIMPLE_MODELS`, `AVAILABLE_CHAT_TEMPLATE_MODELS`) declare available models, merged into `ModelManifest` objects at startup. Chat is always preferred over simple unless `force_simple=True`. (#1070)
- **Unified OpenAI model naming**: canonical names shortened from `openai_compatible` / `async_openai_compatible` to `openai` / `async_openai`. Old names continue to work as aliases via `MODEL_ALIASES`. File renames: `chat/openai_compatible.py` -> `chat/openai.py`, `simple/openai_compatible.py` -> `simple/openai.py`. (#1083, #1084)
- **Simple mode deprecation**: the `doc_to_visual` + `doc_to_text` interface for API models is deprecated. New integrations should use `doc_to_messages` + `ChatMessages`.

### Statistical Analysis

- **CLT and clustered standard error estimation**: for benchmarks with correlated questions (e.g., multiple questions per video), specify `cluster_key` in task YAML to apply cluster-robust SE correction. Clustered SE can be 3x larger than naive estimates. Output format: `score +/- 1.96 x SE` (95% CI). (#989)
- **Baseline-anchored paired comparison**: paired t-test on per-question differences `d_i = score_A - score_B`, removing question difficulty variance to isolate model difference signal. Reports `mean_diff`, CI, and p-value. (#1006)
- **Power analysis**: compute minimum sample size to detect a given effect size (e.g., 2% improvement) before running an evaluation. Rule of thumb: reliable benchmarks need `n > 1000`. (#1007)
- **Model stability metrics**: run N samples per question (temp=0.7), report expected accuracy (EA), consensus accuracy (CA), internal variance (IV), and consistency rate (CR).
- **Decontamination probing**: settings for detecting potential data contamination in video benchmarks. (#990)

### Evaluation as a Service

- **HTTP eval server**: FastAPI-based server with endpoints for job submission (`/evaluate`), status polling (`/jobs/{id}`), queue management (`/queue`), and resource discovery (`/tasks`, `/models`). Includes `JobScheduler` for sequential GPU resource management. (#972)
- **Client libraries**: `EvalClient` (sync) and `AsyncEvalClient` (async) for programmatic job submission from training loops.
- **Web UI**: React + FastAPI web interface replacing the terminal TUI, with model/task selection, real-time command preview, and live output streaming. (#1001)

### New Tasks

**Spatial & 3D reasoning**:
- 3DSR (#1072), Spatial457 (#1031), SpatialTreeBench (#994), ViewSpatial (#983), OmniSpatial (#896)
- SiteBench (#984, multi-image #996), VSIBench (debiased & pruned #975, multi-image #993)
- Blink, CV_Bench, Embspatial, ERQA (#927), RefSpatial, Where2Place (#940)
- SpatialViz (#894)

**Knowledge & reasoning**:
- CoreCognition (#1064), MMSU (#1058), Uni-MMMU (#1029), Geometry3K (#1030)
- AuxSolidMath (#1034), MindCube (#876), MMVP (#1028), RealUnify (#1033)
- IllusionBench (#1035), MME-SCI (#878), VLMs are Blind (#931), VLMs are Biased (#928)
- Reasoning task versions for multiple benchmarks (#926, #1038)
- VLMEvalKit-compatible Qwen task variants for MMMU and MMStar (#1021)

**Video & streaming**:
- MMSI-Video-Bench (#1053), OVOBench (#957), Mantis-Eval (#978)
- LongVT for long video with tool calling (#944), SciVideoBench (#875)

**Multimodal & other**:
- PRISMM-Bench (#1063), OSI-bench (#1068), mmar (#1057), PAIBench-U (#1050)
- SPAR-bench (#1011), BabyVision Gen (#1010) + Und (#1015)
- AV-SpeakerBench (#943), imgedit bench (#941), MMSearch-Plus (#1054)
- CaptionQA (#1004), StructEditBench (#1016), kris_bench (#1017)
- FALCON-Bench (#942), UEval (#890), SeePhys (#903), SNSBench (#930)
- STARE (#893), GroundingMe (#949), GEditBench (#939), JMMMU-Pro (#937)
- WenetSpeech test_net split (#1027)

### New Models

- **GLM4V, LLaMA 4** (#1056)
- **OmniVinci, MiniCPM-o-2_6** (#1060)
- **Uni-MoE-2.0-Omni, Baichuan-Omni-1d5** (#1059)
- **Audio Flamingo 3, Kimi Audio** (#1055)
- **InternVL-HF** (#1039), **InternVL3, InternVL3.5** (#963)
- **Bagel UMM** (#1012), **Cambrian-S** (#977)
- **Qwen3-VL** (#883), **Qwen3-Omni, Video-Salmonn-2** (#955)
- **LLaVA-OneVision-1.5** chat interface (#887)
- **Multi-round generation** (`generate_until_multi_round`) for Qwen2.5-VL and Qwen2-VL (#960)

### Bug Fixes

- Raise minimum supported Python version to 3.10 (#1079)
- Fix video loader memory leaks via resource cleanup (#1026)
- Replace hardcoded `.cuda()` with `.to(self._device)` for multi-GPU support (#1024)
- Fix Qwen2.5-VL nframes edge case (#992, #987)
- Fix multi-image token insertion for Cambrians model (#1075)
- Add dynamic `max_num` calculation to InternVL3 (#1069)
- Fix `partial` support in VSIBench metric calculation (#1041)
- Fix Qwen2-Audio parameter name error (#1081)
- Fix InternVL3 duplicate `<image>` token issue (#999)
- Fix hallusionbench processing for distributed eval (#885)
- Fix COCO Karpathy test data loading (#884)
- Fix nested dictionary input for vLLM `mm_processor_kwargs` (#915)
- Fix log_samples missing fields in doc (#731)
- Fix catastrophic backtracking in Charades eval regex
- Filter multimodal content from log samples while preserving metadata (#962)
- Fix Qwen2.5-VL batch size > 1 visual alignment (#971)

### Infrastructure & Documentation

- Developer guidance for AI agents and contributors (AGENTS.md) (#1085)
- Restructured v0.6 release notes: top-down architecture overview
- README: reordered sections by user journey, simplified header
- Added CITATION.cff, FAQ, quickstart guide
- i18n README translations for 18 languages (#979)
- Scalable choice selection for evaluation (#1005)
- Use dependency lower bounds for broader compatibility (#969)

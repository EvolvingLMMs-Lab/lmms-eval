# Changelog

## v0.7 (2026-02-27)

[Release notes](lmms-eval-0.7.md)

### Highlights

- **25+ new benchmark tasks** spanning document understanding, video, math, spatial, AGI, audio, and safety domains. ([§1](lmms-eval-0.7.md#1-new-benchmark-tasks))
- **Image/Video I/O throughput upgrade**: unified `read_video` with TorchCodec backend (up to **3.58x** faster), DALI GPU decode, shared encoding helpers, and LRU caching. ([§3](lmms-eval-0.7.md#3-imagevideo-io-throughput-upgrade))
- **Lance-backed video mode**: single Lance table on Hugging Face. Local-file, Lance-blob, and YouTube-URL resolution with priority fallback. ([§4](lmms-eval-0.7.md#4-lance-backed-video-mode))
- **Safety and red-teaming baseline**: `safety_redteam` group with jailbreak ASR, refusal rate, toxicity, and over-refusal metrics. ([§5](lmms-eval-0.7.md#5-safety-and-red-teaming-baseline))
- **Efficiency metrics coverage**: per-sample token counts, run-level throughput, and TTFT/TPOT on vLLM backends. ([§6](lmms-eval-0.7.md#6-efficiency-metrics-coverage))
- **Agentic task evaluation**: new `generate_until_agentic` output type with iterative tool-call loop, deterministic simulators, and trace-level metrics. ([§8](lmms-eval-0.7.md#8-agentic-task-evaluation))
- **YAML config-driven evaluation**: single `--config` YAML file replaces fragile CLI one-liners. Schema validation, env expansion, batch configs, full reproducibility via `resolved_cli_args`. ([§9](lmms-eval-0.7.md#9-better-one-line-evaluation-support))
- **Pipeline-level reasoning tag stripping**: `<think>` blocks stripped in `evaluator.py` after filters, before `process_results()`. All backends, configurable via `--reasoning_tags`. ([§10](lmms-eval-0.7.md#10-pipeline-level-reasoning-tag-stripping))
- **Async OpenAI `message_format`**: replaces `is_qwen3_vl` flag and `async_openai_qwen3_vl` class. `generate_until()` decomposed into 7 focused methods. ([§11](lmms-eval-0.7.md#11-support-customized-message_format-in-async_openai))
- **Flattened JSONL log output**: `generate_until` responses flattened from doubly-nested to singly-nested lists. Multi-choice tasks unchanged. ([§12](lmms-eval-0.7.md#12-flattened-jsonl-log-output))

### New Benchmark Tasks

**Document understanding**:
- OmniDocBench ([#1152](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1152)), MMLongBench ([#1169](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1169)), MMLongBench-Doc ([#1164](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1164)), DUDE ([#1151](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1151)), OfficeQA ([#1150](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1150))

**Video**:
- Neptune long-video benchmarks ([#1187](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1187)), TVBench ([#1160](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1160)), ViVerBench ([#1166](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1166)), EgoTempo ([#1155](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1155))

**Math & reasoning**:
- MathCanvas ([#1161](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1161)), MathKangaroo ([#1158](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1158)), VisuLogic ([#1159](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1159)), LLaVA-OV 1.5 RL reasoning collection ([#1208](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1208))

**Spatial & counting**:
- Point-Bench ([#1157](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1157)), CountBench ([#1156](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1156)), FSC-147 ([#1163](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1163))

**Knowledge & QA**:
- SimpleVQA ([#1184](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1184)), WorldVQA ([#1168](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1168)), MTVQA ([#1167](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1167)), HiPhO ([#1186](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1186)), MME-CC ([#1185](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1185)), VPCT ([#1183](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1183)), ZeroBench ([#1182](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1182))

**AGI & agentic**:
- ARC-AGI-1, ARC-AGI-2, BrowseComp ([#1190](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1190)), Vending-Bench 2, τ2-Bench Telecom

**Audio** (v0.7 Audio Update, [#1124](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1124)):
- AMI (meeting transcription, train/validation/test splits)
- CN College Listen MCQ (Chinese college listening comprehension)
- DREAM TTS MCQ (dialogue-based listening comprehension)
- EuroPal ASR (European Parliament speech recognition, test/validation splits)
- Song Describer (music captioning, train/validation splits)

**Safety**:
- JailbreakBench harmful + benign splits (`safety_redteam` group)

### New Models

- **NanoVLM** (SigLIP2 + MLP projector + Qwen3-0.6B): chat-style evaluation with async multi-GPU inference via job-queue dispatch. ([#1207](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1207))
- **Async HF model**: generic async multi-GPU worker backend for HuggingFace model families. Loads replicas on N GPUs with independent worker threads. ([#1204](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1204))

### Image/Video I/O Throughput Upgrade

- **Unified `read_video` entry point**: single function in `load_video.py` with `backend` parameter and `LMMS_VIDEO_DECODE_BACKEND` env var. Supports PyAV (default), TorchCodec, and DALI (GPU). ([Release notes §3](lmms-eval-0.7.md#3-imagevideo-io-throughput-upgrade))
- **TorchCodec multi-threaded decode**: up to **3.58x** faster than PyAV at 8 frames with `LMMS_VIDEO_TORCHCODEC_THREADS=8`. Thread tuning required — default threads (0/1) regress up to +150%.
- **DALI GPU decode**: optional GPU-accelerated backend via `LMMS_VIDEO_DALI_DEVICE=gpu`.
- **Shared image encoding**: `encode_image_to_base64` consolidated across protocol and simple adapters with path-metadata keyed cache.
- **Media resolver LRU caching**: `_candidate_roots_cached(maxsize=256)` and `_extension_variants(maxsize=4096)` cache deterministic path-derivation work. `Path.exists()` checks remain uncached for correctness.
- **PyAV stream fallback**: `seek(0)` before packet decode on stream failure. Set-membership lookup in frame selection. Preallocated output array fill replaces `list` + `np.stack`.
- **LongVideoBench validation**: decode latency `2.79s` -> `1.02s` (**-63%**, 2.7x speedup) with score reproducibility confirmed.

### Lance-Backed Video Mode

- **Lance table distribution**: MINERVA videos stored as `video_blob` in a Lance table on Hugging Face (`lmms-lab-eval/minerva`). Single-package reproducible distribution. ([Release notes §4](lmms-eval-0.7.md#4-lance-backed-video-mode))
- **Resolution priority**: local file (`MINERVA_VIDEO_DIR`) > Lance blob (`MINERVA_LANCE_VIDEO_URI`) > YouTube URL fallback.
- **Build tooling**: `tools/minerva_to_lance.py` converts local video files to Lance tables. `tools/bench_minerva_video_resolution.py` and `tools/bench_minerva_pipeline_latency.py` for latency benchmarking.

### Safety and Red-Teaming Baseline

- **`safety_redteam` task group**: `safety_jailbreakbench_harmful` and `safety_jailbreakbench_benign` from `JailbreakBench/JBB-Behaviors`. ([Release notes §5](lmms-eval-0.7.md#5-safety-and-red-teaming-baseline))
- **Harmful split metrics**: jailbreak ASR, refusal rate, toxicity score, content filter rejection rate, demographic/non-demographic refusal rates.
- **Benign split metrics**: over-refusal rate, benign toxicity score, content filter rejection rate, demographic/non-demographic refusal rates.
- **Dual toxicity backends**: Perspective API when `PERSPECTIVE_API_KEY` is set; offline keyword heuristic fallback.

### Efficiency Metrics Coverage

- **Per-sample token counts**: `input_tokens`, `output_tokens`, `reasoning_tokens` (when backend metadata exists) in evaluation outputs. ([Release notes §6](lmms-eval-0.7.md#6-efficiency-metrics-coverage)) ([#1125](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1125))
- **Run-level throughput**: `total_gen_tokens`, `total_elapsed_time`, `avg_speed` in results JSON.
- **TTFT/TPOT**: available on `vllm` chat backends via native runtime metrics. Not available on SGLang, OpenAI API, or HuggingFace local backends (wall-clock throughput only).
- **Token-based efficiency output**: `efficiency.overall.tokens_per_correct_answer`, `efficiency.overall.avg_output_tokens_per_sample`, and per-task breakdown. No provider-specific pricing dependencies.

### Skill-Based Agent Workflows

- **Repository skill**: `skills/lmms-eval-guide/SKILL.md` with reference files for models, tasks, and API server integration. Turns lmms-eval into a reusable operational skill for coding agents. ([Release notes §7](lmms-eval-0.7.md#7-skill-based-agent-workflows))
- **Dispatch routing**: model/task extension via skill references; training integration via HTTP service; quick validation via `--limit` smoke tests.

### Agentic Task Evaluation

- **`generate_until_agentic` output type**: iterative evaluator loop where the model emits `<tool_call>` or `<submit>` tags and `doc_to_text` executes tools against deterministic Python simulators. Configurable via `max_agentic_steps`. ([Release notes §8](lmms-eval-0.7.md#8-agentic-task-evaluation))
- **Agentic tasks**: `vending_bench2` (vending machine operation, 4 tools, 10 steps) and `tau2_bench_telecom` (telecom support, 4 tools, 8 steps). No external sandbox required.
- **Trace metrics**: success rate, trace step validity, state progress, termination quality, and composite trace quality.

### Better One-Line Evaluation Support

- **`--config` flag**: one YAML file replaces long CLI commands. Maps directly to CLI arguments plus an `env` section for credentials and paths. ([Release notes §9](lmms-eval-0.7.md#9-better-one-line-evaluation-support))
- **Environment variable expansion**: `${VAR}` expands from shell; `${VAR:-default}` provides fallback. Keys containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` are auto-masked in log output.
- **Override priority**: defaults < YAML < CLI. CLI arguments always win, enabling a canonical config with per-run overrides.
- **Schema validation**: unknown keys in YAML now raise an error listing valid keys. Typos like `modle` no longer silently pass.
- **Batch evaluation**: YAML accepts a list of configs for multi-model runs in a single invocation.
- **Full reproducibility**: results JSON includes `resolved_cli_args` — the complete merged configuration. Reconstruct the exact experiment from any results file.
- **Web UI integration**: export/import YAML configs from the Web UI. Round-trip is lossless.

### Pipeline-Level Reasoning Tag Stripping

- **Pipeline-level stripping** (`lmms_eval/api/reasoning.py`): `strip_reasoning_tags()` runs in `evaluator.py` after the filter pipeline and before `process_results()`. All model backends benefit without per-model changes. ([Release notes §10](lmms-eval-0.7.md#10-pipeline-level-reasoning-tag-stripping))
- **Dual output preservation**: `resps` stores raw output (with `<think>` blocks) for chain-of-thought analysis; `filtered_resps` stores the clean scored text. `resps` omitted when identical to `filtered_resps`.
- **CLI control**: `--reasoning_tags` accepts `none` (disable), default `<think>...</think>`, or custom JSON pairs like `'[["<think>", "</think>"], ["<reasoning>", "</reasoning>"]]'`.
- **Per-task override**: set `reasoning_tags` in task YAML to override the CLI flag per-task.
- **Removed ad-hoc handling**: deleted `parse_reasoning_model_answer` calls from 6 model files.

### Support customized message_format in async_openai

- **`message_format` parameter**: replaces `is_qwen3_vl` flag and the separate `async_openai_qwen3_vl` model class. Currently supports `openai` (default) and `qwen3_vl` (per-frame timestamps for video). New formats require only an `elif` in `prepare_messages()`. ([Release notes §11](lmms-eval-0.7.md#11-support-customized-message_format-in-async_openai))
- **Decomposed `generate_until()`**: 130-line monolith split into `_build_video_kwargs()`, `prepare_messages()`, `_get_initial_concurrency()`, `_compute_dispatch_order()`, `_process_with_retry()`, `_update_concurrency()`, `_run_scheduling_loop()`. Main method is now 8 lines.
- **`_AdaptiveConcurrencyTracker`**: concurrency tracking state moved from scattered `nonlocal` closures to a dedicated dataclass.

### Flattened JSONL Log Output

- **Single-instance flattening**: `generate_until` responses flattened from `[["text"]]` to `["text"]` at serialization time. Multi-choice `loglikelihood` tasks with N instances per doc remain `[["a"], ["b"], ...]`. ([Release notes §12](lmms-eval-0.7.md#12-flattened-jsonl-log-output))
- **Dedup preserved**: `resps` omitted when identical to `filtered_resps` after flattening.
- **In-memory format unchanged**: `logged_samples` retains nested format for existing consumers (wandb logger, etc.).

### CLI & UX

- **Subcommand architecture** (`lmms_eval/cli/`): modular dispatch with `eval`, `tasks`, `models`, `ui`, `serve`, `power`, `version` subcommands. Interactive wizard when `eval` is invoked without arguments. ([#1202](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1202))
- **Branded evaluation banner**: startup banner showing version metadata and environment info.
- **External usage guide**: new `docs/external_usage.md` covering CLI subcommands and Python library access.

### Bug Fixes

- Fix image vs video file path detection in `auto_doc_to_messages` fallback
- Align `osworld_g` polygon scoring with osworld-verified annotations ([#1165](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1165))
- Harden `mmsi-bench` utils parsing against malformed model responses ([#1162](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1162))
- Fix Whisper `world_size` initialization for single-process runtime ([#1124](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1124))
- Fix Audio Flamingo 3 parameter handling ([#1124](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1124))
- Restore `task_input_specs/redundancy_refactor.yaml` accidentally deleted by test commit

### Refactoring

- **Test suite modernization**: migrated all test files from `unittest.TestCase` to pure pytest style (`test_protocol`, `test_construct_requests`, `test_evaluator`, `test_task_pipeline`, CLI tests). Added prompt stability tests and benchmark registration coverage.
- **Video loader cleanup**: renamed `read_video_pyav` -> `read_video` with backward-compat alias; removed dead `read_video_pyav_pil`, `read_video_pyav_base64`, and `_resize_image` functions; updated all 12 caller files.
- **Reasoning utils deduplication**: consolidated repeated reasoning utility patterns into factory functions across task modules.
- Removed `use_custom_video_loader` dead code from 5 model files (qwen2_5_vl, qwen3_vl, qwen3_omni, llava_onevision1_5, huggingface).
- **Async OpenAI internal decomposition**: `generate_until()` split into 7 focused methods with `_AdaptiveConcurrencyTracker` dataclass.

### Infrastructure & Documentation

- Comprehensive test suite README with concrete code examples for each test category.
- Rewritten `docs/README.md` with pipeline diagram, code examples, and categorized table of contents.
- External usage guide (`docs/external_usage.md`) for CLI subcommands and Python library API.
- Updated example scripts: switched to OpenAI API model, promoted MMMU/VideoMMU/LongVideoBench as recommended benchmarks.
- SpatialTreeBench naming variant documentation ([#1189](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1189)).
- Agent skill files (`skills/lmms-eval-guide/`) for model, task, and API server integration workflows.
- Example YAML configs (`configs/`) for local, API, and batch evaluation patterns.

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

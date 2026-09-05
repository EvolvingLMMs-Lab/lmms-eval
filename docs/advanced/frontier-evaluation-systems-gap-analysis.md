# Frontier Evaluation Systems: Capability Gap Analysis for lmms-eval

Evidence date: 2026-09-01  
lmms-eval snapshot: `787d0573139569027b8c879433792d38f1f2cd74` (`origin/main`)  
Scope: public primary sources plus `git show` and `git grep` against the pinned upstream tree. The original audit did not execute evaluations.

## September 5 Recheck and Delivery Decision

The recheck used upstream `e0bfc699` and live pull-request state on September 5, 2026. The direction remains useful, with the corrections below. This section supersedes conflicting priorities or claims in the original audit.

1. The repository already has a detailed modular pipeline RFC in PR #1439. It defines EvalSpecV1, EvalResultV1, artifact publication, ownership, and frontend adapters. Continue that design instead of introducing a competing RunBundle or request hierarchy.
2. Several capabilities already have delivery owners. CPU contracts are in #1440, registry changes in #1446 through #1450, TUI security in #1442 through #1445, and performance/provenance in #1452 through #1460. Agent environments are in #1418/#1419, and group postprocessing is in #1499. These PRs were open when checked; implementation on a branch does not imply availability on main.
3. The original cache assessment was too generous. Live writes reject empty answers, but audit replay did not apply that validator. Reopening a cache could restore rejected answers. Existing invalid SQLite entries also bypassed validation on lookup. This is the first bounded correctness fix, including recovery from valid audit entries and repair during shard merging.
4. The four explicit cluster_key declarations are a text-search count, not effective task coverage. YAML inheritance must be resolved before reporting adoption. A paired t-test is not automatically incorrect for binary observations; comparison identity, observation units, missingness, and repeated-trial semantics deserve review before adding more tests.
5. Contamination support is partial. The generic filter is a stub, but video tasks already provide no-visual, option-shuffling, and related probes. These probes can expose shortcuts; they do not prove training-set overlap. A generic detector needs a specified reference corpus and a defined interpretation of its evidence.
6. Verifiers flag judge failures in metadata, but return a numeric zero. Tasks must propagate that flag for the final denominator to distinguish infrastructure failure from a wrong answer. The original wording overstated automatic separation.
7. Full sandboxing, multi-worker scheduling, new executor backends, and calibrated multi-judge decisions remain candidate features. They need concrete callers and validation data. The existing RFC deliberately defers a general sandbox and multi-worker execution.
8. Inference/scoring separation is incomplete on main. The evaluator collects responses before scoring; final sample files are written after evaluation returns. Optional response-cache logs help recovery, but are not an unconditional durable scoring artifact. The legacy from_log reader expects an older JSON shape rather than the current sample JSONL contract.

The checked core files were unchanged between `787d0573` and `e0bfc699`: HTTP scheduling/protocol, Instance, ChatMessages, ModelRegistryV2, response cache, statistical metrics, result writer, and decontamination filter. This confirms those source findings still apply without claiming production behavior was tested.

The delivery order is:

| Step | Work | Evidence needed |
| --- | --- | --- |
| First | Repair response-cache recovery and reads in a small independent fix. | Invalid legacy audit/SQLite rows never become hits; valid responses survive restart; successful retries repair shared entries; existing cache tests pass. |
| Existing foundations | Continue the open CPU-contract, registry, security, and provenance PRs under their current owners. | Their own tests, checks, and reviews pass before landing. |
| Frozen evaluation contract | Implement the next eligible slice of #1439, preserving its Instance and artifact boundaries. | Frontend parity, source identity, and deterministic projection tests pass. |
| Reuse and decisions | Add offline scoring and comparison over the established artifact contract. | Scoring requires no primary inference; sample identities and denominators match; decision thresholds are explicit. |
| Extensions | Add judge calibration, additional executors, or realtime media when a concrete workflow requires them. | A representative end-to-end use case and its acceptance criteria are specified. |

The first fix is published as [draft PR #1513](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1513), commit `43aa10e6421970a1677543c0f0397ee0de75fc1f`. The cache suite passed 61 tests and four subtests. Six new recovery tests produce nine assertion failures against the base cache module, including subtests. Full-repository pre-commit passed, and independent correctness, silent-failure, and ML reviews found no remaining issues after shared-cache repair was added. Validation used mocked models and temporary cache files.

While this fix was being prepared, other delivery work advanced main to `4eb7d01b`: #1440, #1451, #1445, and #1441 landed. Those merges were not performed by this recheck. The initial open-PR inventory above records the earlier observation; CPU contracts, the TUI security collector, and the per-task sample-limit fix are now on main.

Rechecked reference patterns: [Inspect eval sets](https://inspect.aisi.org.uk/eval-sets.html), [Inspect model grading](https://inspect.aisi.org.uk/model-graded.html), [NeMo result format](https://docs.nvidia.com/nemo/evaluator/evaluation/result-format), and [HELM maintenance notice](https://github.com/stanford-crfm/helm). These support the capability comparison, not a requirement to reproduce every peer feature.

Delivery references: [pipeline RFC #1439](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1439), [CPU contracts #1440](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1440), [agent environment #1419](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1419), and [group postprocessing #1499](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1499).

## Executive finding

lmms-eval is already strong where many evaluation frameworks are weak: multimodal benchmark coverage, model adapters, crash-aware response caching, per-sample records, and statistical reporting. The next differentiator should not be another wave of benchmark and backend integrations. It should be a trustworthy evaluation runtime with explicit contracts.

The three highest-value improvements are:

1. A strict, versioned `EvalSpec` that compiles into a typed execution plan and performs model/modality/capability validation before any dataset download or model load.
2. A versioned, append-only `RunBundle` that makes every score replayable and auditable, including dataset/media identity, model and judge identity, code and environment identity, failures, and partial-run state.
3. A real trust boundary for task code, media, tools, and the HTTP service, with sandboxed execution and explicit network/filesystem/secret authority.

After those foundations, lmms-eval should make agent/tool trajectories, durable multi-backend scheduling, judge reliability, and contamination analysis first-class. These are the areas where Inspect AI, NeMo Evaluator, SWE-bench, and tau2-bench now expose architectural patterns that lmms-eval can reuse without copying their product scope.

Status terms in this note are deliberate:

- **Found**: a generic framework capability is visible in the inspected source.
- **Partial**: useful machinery exists, but the public contract is incomplete or task/backend-specific.
- **Not found in scoped audit**: the searches and core paths inspected did not reveal a generic implementation. This is not a claim that no task-specific implementation exists anywhere in the repository.

## What lmms-eval already has

| Area | Current evidence | Assessment |
| --- | --- | --- |
| Task configuration | [`TaskConfig`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/api/task.py#L101-L160) declares the task surface and validates the output type during construction. | Useful declarative surface, but Python annotations are not a strict serialized schema and many nested values remain untyped dictionaries. |
| Runtime request contract | [`Instance`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/api/instance.py#L74-L101) has a typed request name, but carries an untyped positional `arguments: tuple`; generation kwargs are recovered by scanning for the only dictionary in that tuple. | This is the most fragile internal seam. New task and model paths must agree on tuple order implicitly. |
| Model extension | [`ModelManifest`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/models/registry_v2.py#L11-L48) and Python entry-point loading provide a clean model registration seam with alias and chat/simple validation. | Strong base. The manifest does not yet declare modalities, tools, streaming, logprobs, structured output, dependency versions, or compatibility ranges. |
| Multimodal protocol | [`ChatTextContent`, `ChatImageContent`, `ChatVideoContent`, and `ChatAudioContent`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/protocol.py#L16-L47) provide structured messages and adapters for provider formats. | Strong start. Roles are limited to user/system/assistant; media references are `Any`; document/PDF, tool/result messages, and typed multimodal output are missing; media authority, MIME, digest, transform, and sampling identity are not part of the contract. |
| Agentic and tool evaluation | [`generate_until_agentic`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/evaluator.py#L810-L960) implements bounded rounds and optional full trace capture; two task YAMLs select it. MCP tool loops also exist inside the [`async_openai`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/models/chat/async_openai.py#L166-L296) and [`sglang`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/models/chat/sglang.py#L374-L470) backends. | Partial. Environment state and task tools are callback dictionaries, while MCP execution is backend-owned; neither path emits one central typed, replayable, policy-governed trajectory. |
| Response reuse | [`ResponseCache`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/caching/response_cache.py#L374-L496) uses SQLite plus an fsynced JSONL audit log, validates deterministic requests, fingerprints model/task/config fields, isolates rank writes, and merges shards. | A real strength. The cache key does not bind dataset revision or media bytes, uses a fixed generation-key allowlist, and does not restore an agent environment or a whole suite execution plan. |
| Statistical analysis | The framework implements bootstrap error, clustered standard error, repeated-sample stability, paired comparison, and power analysis in [`metrics.py`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/api/metrics.py#L763-L935) and [`evaluator_utils.py`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/evaluator_utils.py#L115-L210). | Stronger than most peer defaults. Binary paired comparisons still use a t-test; generic exact McNemar tests, paired bootstrap/permutation tests, multiple-comparison correction, and suite decision policies were not found. Only four of 1,928 task `.yaml` files declare `cluster_key`, so framework support is much broader than current task adoption. |
| Judge abstraction | [`VerifyResult`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/verifiers/base.py#L16-L36), composite fallback, and OpenAI/Gemini verifier implementations separate judge failures from wrong answers and retain raw output. | Useful base. Generic replicated/multi-judge grading, position swapping, calibration sets, agreement metrics, and prompt-injection hardening were not found. |
| Execution surfaces | The evaluator supports Accelerate and torchrun. The HTTP [`EvaluateRequest`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/entrypoints/protocol.py#L24-L38) exposes only a subset of CLI options, while [`JobScheduler`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/entrypoints/job_scheduler.py#L68-L210) launches subprocesses through a sequential in-memory queue and refuses to cancel a running job. | Partial. CLI, HTTP, and MCP can resolve different effective contracts. There is no common durable executor contract across local, Slurm, Kubernetes/Ray, or provider batch systems. |
| Results and provenance | Results include resolved CLI arguments, seeds, git hash/branch, lmms-eval version, task configs/versions, usage, throughput, sample hashes, and per-sample JSONL records in [`evaluator.py`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/evaluator.py#L681-L744) and [`evaluation_tracker.py`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/loggers/evaluation_tracker.py#L169-L300). | Good information, but no result schema version or validator was found. Environment capture is commented out, model/dataset revisions are not guaranteed, and final files are ordinary writes rather than one append-only run log with atomic finalization. Prompt-drift snapshots cover eight benchmarks, a valuable control with narrow catalog coverage. |
| Integrity preflight | `--check_integrity` calls [`run_task_tests`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/utils.py#L742-L776), which searches for `tests/test_version_stable.py`; the pinned tree contains `test/` and no such file. | **Static inference:** the advertised integrity path appears unable to reach a test suite at this SHA. This should be verified with a direct CLI repro, then fixed before it is used as a trust claim. |
| Contamination | Task fields exist and one of 1,928 task `.yaml` files enables them, but the generic [`DecontaminationFilter.apply`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/filters/decontamination.py#L4-L23) is a stub. | Not implemented at framework level in the inspected path. |
| Trust boundary | Full task YAML loading imports `!function` code, external include paths are supported, and one dataset path invokes an interpolated command with `shell=True` in [`Task.download`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/787d0573139569027b8c879433792d38f1f2cd74/lmms_eval/api/task.py#L961-L986). Backend-specific MCP and Thyme restricted-execution paths exist, but the HTTP server explicitly says it is for trusted environments only. | A generic evaluator-wide trust model is not found. The trusted-code assumption is too broad for a hosted evaluator or third-party task marketplace. |

## Patterns from current evaluation systems

The point of this comparison is not to choose a winner. Each system makes a different trade-off, and the reusable ideas are narrower than the products themselves.

| System | Capability pattern worth extracting | Relevance to lmms-eval |
| --- | --- | --- |
| OpenAI Evals and the OpenAI Evals API | The open-source registry uses versioned eval identifiers and JSONL data/templates; the current platform API goes further by requiring an item JSON Schema and exposing typed string, similarity, model, Python, and composite graders. [1][2][3] | Treat the eval definition, input row, sample output, and grader as separately versioned schemas. Do not infer their shapes from task callbacks at runtime. |
| Inspect AI | A task composes a dataset, solver, and scorer over a typed `TaskState`; solvers and agents share model/tool state. Inspect also has provider adapters, typed multimodal content, sandboxes, tool approval, resumable eval sets, recoverable versioned logs, extension entry points, offline rescoring, transcript scanners, and multi-judge agreement metrics. [4][5][6][7][8][9][10][11][12][13] | This is the clearest reference for a unified static-plus-agent evaluation protocol. The most valuable ideas are the typed state/event log, authority-aware media, execution policy, and the separation of generation from scoring. |
| EleutherAI lm-evaluation-harness | Shareable task YAML plus a code commit remains a simple reproducibility unit. It also has a functioning n-gram decontamination pipeline that reports clean metric variants. [14][15] | Keep lmms-eval's low-friction YAML authoring, but finish the contamination contract and place strict validation around embedded code. |
| HELM | HELM separates serializable specifications and states from controllers: Scenario -> Adapter -> Executor -> Metric, driven by a `RunSpec`. It also treats accuracy, efficiency, bias, toxicity, prompt inspection, and leaderboard publication as one suite. [16][17] | A serializable plan/state/controller separation would reduce coupling in lmms-eval. HELM entered maintenance mode in 2026, so the architecture is a reference, not a platform dependency. |
| Hugging Face LightEval | Typed `LightevalTaskConfig`, `Doc`, and `ModelResponse` objects feed sample- and corpus-level metrics. It supports multiple inference backends/parallelism modes and writes detailed Parquet records through fsspec-compatible output paths. [18][19][20] | Typed request/response objects and columnar details would make post-hoc analysis and cross-run comparison more reliable without requiring a heavier service. |
| NVIDIA NeMo Evaluator | Environments, solvers, and executors are separate protocols; official docs cover local, Docker, Slurm, Kubernetes, Ray, sharding, checkpoint/resume, trajectory files, result exporters, paired run comparison, and quality gates. Current comparison uses per-problem identity and McNemar-style reasoning for binary outcomes. [21][22][23][24] | lmms-eval can retain its evaluator core while adding executor and result-bundle interfaces. NeMo also demonstrates how to host other harnesses behind URI-addressed environments instead of absorbing their code. |
| MLCommons MLPerf Client | A scenario JSON selects models and vendor execution providers; assets can be verified by SHA-256, runs can be made offline, vendor code can be subprocess-isolated, and benchmark components are governed as base/extended/experimental with quality qualification. [25][26] | Useful for plugin governance, artifact integrity, offline reproducibility, and leaderboard qualification. This is a performance benchmark rather than a general model-quality harness, so only those contracts transfer. |
| VLMEvalKit | The toolkit standardizes interleaved multimodal messages, makes video `fps`/`nframe` choices explicit, provides a model/dataset config matrix, supports torchrun-style distributed inference, and retains inspectable inference and judge artifacts. [27][28][29] | lmms-eval should make media preprocessing identity and judge identity first-class, not merely backend/task kwargs. Compatibility should be defined at an artifact/protocol boundary rather than task-specific score matching. |
| SWE-bench and SWE-agent | SWE-bench evaluates each patch in a container, defines a small prediction schema, preserves per-instance logs, and exposes layered image caching. SWE-agent uses strict Pydantic configs, deployment abstractions, replay configuration, and typed trajectories. [30][31][32] | Per-sample isolation, replayable trajectories, and layered environment caches are directly applicable to code, computer-use, and other stateful agent evaluations. |
| tau2-bench | A domain owns policy, tools, tasks, environment state, and a user simulator. The framework records trajectories, can re-evaluate them independently, reports repeated-trial reliability, and supports both half-duplex text and full-duplex voice interaction. [33][34][35] | Agent success should be verified from final state and policy compliance, with simulator/environment failures separated from model failures. A trajectory must remain scoreable after generation. |

The current issue tracker points in the same direction: open requests ask for non-scalar group post-processing, reasoning/evaluation decoupling, and separating or parallelizing judge work. These are user demand signals, not proof that every manual workaround is absent. [36][37][38]

## Gap matrix

| Capability | lmms-eval status | Concrete gap | Priority and opportunity |
| --- | --- | --- | --- |
| Strict typed eval specification | Partial | `TaskConfig` is annotated, but nested config and the request ABI remain dictionaries/tuples. There is no standalone schema that another tool can validate or generate, and static inspection indicates the existing `--check_integrity` target path is stale. | **P0**: repair the current integrity command, then introduce `EvalSpecV1` with discriminated task, model, scorer, media, limits, and execution policies plus `validate` and `plan --json`. |
| Provider capability negotiation | Partial | Model manifests identify class paths and chat/simple mode, not modality, tools, logprobs, structured output, streaming, batching, or limits. Unsupported combinations often fail after loading. | **P0**: add a capability manifest and compile-time requirement matching. Preserve provider-specific kwargs only behind the adapter boundary. |
| Multimodal asset protocol | Partial | Image/video/audio exist, but references are untyped, PDF/document is missing, and MIME, digest, frame sampling, decoder/transform identity, and filesystem/network authority are not represented. | **P0**: add `MediaRef` and `MediaTransformSpec`; hash bytes or immutable object identity into cache/run IDs; reject unsupported media before inference. |
| Reproducibility and provenance | Partial | Seeds, code hash, configs, and task hashes are recorded, but model weight revision, dataset revision/content manifest, environment lock/container digest, media preprocessing, judge version, dirty-tree state, and schema version are not guaranteed. | **P0**: emit a required `RunManifestV1` and provenance completeness check. A leaderboard-eligible run should fail closed when required identities are unresolved. |
| Result schema and replay | Partial | Aggregates and samples are rich dictionaries but have no visible versioned schema. Generation, scoring, and post-hoc edits do not share one event/provenance model. | **P0**: define `RunBundleV1` with append-only events, typed samples/attempts/scores, atomic finalization, offline rescore, and explicit partial/failed status. |
| Isolation and security | Partial | Backend-specific MCP and restricted Python execution paths exist, but built-in and external task code generally execute with evaluator authority; YAML can import Python; media can resolve host paths/URLs; the HTTP service has no built-in production security boundary. | **P0**: define builtin/trusted-extension/untrusted-data modes, sandbox untrusted execution with network off by default, scope secrets, add resource limits and tool approval, and remove shell interpolation. |
| Agent and tool trajectory protocol | Partial | The evaluator loop is bounded and traceable and some backends execute MCP tools, but tool calls/results, environment state, termination, checkpoints, and rewards do not flow through one central typed contract. | **P1**: add typed `Tool`, `ToolCall`, `ToolResult`, `Environment`, `TrajectoryEvent`, and `AgentAdapter` protocols; normalize current MCP paths, state snapshots, replay, external-agent bridges, and state-based verification behind it. |
| Durable scheduling and distributed execution | Partial | Rank-level inference exists, but the HTTP scheduler is sequential and in-memory, running jobs cannot be cancelled, and its request schema omits many CLI controls. No common executor exposes submit/status/cancel/resume across local, Slurm, Kubernetes/Ray, or provider batch APIs. | **P1**: make CLI/HTTP/MCP consume the same `EvalSpec`; separate the plan from executors; persist job state; make stages idempotent; schedule generation, judge, and export independently. |
| Cache, resume, and deduplication | Strong partial | Response reuse and rank merging are robust. Dataset/media identity and all output-affecting provider parameters are not structurally guaranteed; suite and agent state cannot be resumed. | **P1**: derive content-addressed keys from the compiled request, media/dataset digests, adapter version, and provider-normalized config; add suite/sample/agent checkpoints rather than another independent cache. |
| Statistical uncertainty and significance | Strong partial | Bootstrap, CLT/clustered SE, stability, paired t-test, and power are present. Metric observation unit, missingness, test choice, and multiple comparisons are not declared; only four task YAMLs opt into clustered units. | **P1**: add `MetricSpec` fields for unit/cluster/reducer; exact or asymptotic McNemar for paired binary outcomes; paired bootstrap/permutation for continuous metrics; multiple-test correction and `GO/NO-GO/INCONCLUSIVE` gates; migrate correlated-media tasks. |
| Judge calibration and reliability | Partial | Judge providers, raw outputs, failure flags, and rule-first fallback exist. No generic judge replication, order randomization/swap, human calibration set, agreement statistic, or judge-prompt injection defense was found. | **P1**: introduce a fully fingerprinted `JudgeSpec`, offline judge replay, multi-judge/epoch reducers, Krippendorff alpha, position-bias checks, calibration reports, and separated judge-infra denominators. |
| Contamination and leakage | Not found in scoped audit | Config fields exist, one task YAML enables them, and the generic filter is a stub. There is no standard contamination report attached to results. | **P1**: implement pluggable exact/n-gram/embedding/model-based detectors, retain per-sample evidence, and report all/clean/contaminated slices with detector and corpus identity. |
| Observability and failure analysis | Partial | Loguru, progress, W&B/SwanLab, response audit JSONL, token/throughput summaries, and optional agent round traces exist. There is no common event schema, trace/span IDs, lifecycle hook API, or scanner layer for refusal/eval awareness/reward hacking. | **P2**: make the run event stream the source of truth; add hooks/OpenTelemetry export and offline transcript scanners after the event contract is stable. |
| Plugin and version governance | Partial | Model entry points exist, while tasks/metrics/filters mostly rely on imports and mutable registries. Compatibility, capabilities, dependencies, trust level, migrations, and conformance tests are not declared. | **P2**: version an `EvalPluginManifest`, require core API ranges and capability declarations, isolate optional dependencies, and publish conformance fixtures. |
| Leaderboard and submission governance | Partial | Results can be pushed to Hugging Face and dataset cards can be created, but a signed/validated submission bundle and eligibility policy were not found. | **P2**: build validation on `RunBundleV1`; add base/extended/experimental suites, artifact digests, completeness/security gates, and published failure/uncertainty rules. |

## Recommended feature sequence

### P0.1: Compile a strict EvalSpec before execution

The smallest useful cut is a boundary layer, not an immediate rewrite of every task:

```text
YAML / Python task
        -> resolve includes and functions
        -> EvalSpecV1 validation
        -> capability and authority checks
        -> immutable ExecutionPlan + plan_hash
        -> existing evaluator internals
```

The compiled plan should contain:

- Exact task, dataset, split, revision, sample selection, prompt/scorer callable fingerprints, and metric observation units.
- Exact model adapter, model/deployment revision, requested capabilities, normalized generation settings, and resource requirements.
- Exact media references and transformations, including MIME, content/object digest, frame or time sampling, decoder, resize, and detail policy.
- Judge and simulator specs, limits, cache policy, executor policy, and trust/authority policy.

Acceptance gates:

- Unknown fields and invalid unions fail before dataset or model initialization.
- An audio/video/document task cannot be scheduled on an adapter that does not declare support.
- Two hosts resolving the same immutable inputs produce the same `plan_hash`.
- Existing YAML tasks can be translated incrementally, but new public APIs no longer expose positional request tuples.

### P0.2: Make a RunBundle the unit of truth

A minimal durable bundle could be:

```text
run/
  manifest.json       # RunManifestV1 + plan hash + provenance
  events.jsonl        # append-only lifecycle/model/tool/judge/score events
  samples.parquet     # typed final per-attempt records
  results.json        # aggregate view derived from samples
  artifacts/          # large or task-specific referenced outputs
```

Required properties:

- Stable `run_id`, `eval_id`, `sample_id`, `attempt_id`, `request_id`, and parent/baseline IDs.
- A schema version and compatibility/migration policy for every file.
- Status and failure taxonomy that keeps model, provider, dataset/media, judge, simulator, tool, timeout, cancellation, and evaluator failures distinct.
- Secret redaction at record creation, not only at final config serialization.
- Atomic finalization; interrupted runs stay explicitly partial and recoverable.
- Scoring consumes recorded outputs/trajectories, so changing a parser or judge does not repeat primary inference.

This should subsume final-result writing and extend the response audit log. It should not replace the existing response cache with another cache.

### P0.3: Make authority explicit

Recommended modes:

- `builtin`: packaged, reviewed task/model/scorer code.
- `trusted_extension`: pinned installed plugin with declared code execution.
- `untrusted_data`: data-only task using a safe template and built-in transforms/scorers.

For hosted or shared execution:

- Default to `builtin` or `untrusted_data`; require an operator decision for executable extensions.
- Run task tools, Python graders, generated code, and stateful environments in a sandbox with CPU/memory/time/process limits and network disabled unless declared.
- Treat host paths and remote URLs as capabilities; materialize fixed trusted dataset media before model-controlled state exists.
- Scope API keys per model/judge role and never inherit the full server environment into arbitrary task code.
- Add authentication, authorization, rate limits, output-root allowlists, and durable audit IDs before describing the HTTP server as production-ready.

### P1.1: Add a first-class agent/environment/trajectory layer

The protocol should represent messages, model calls, reasoning/output content, tool definitions, tool calls/results, environment observations, state snapshots, approvals, interventions, limits, checkpoints, rewards, and termination reasons as typed events.

Separate interfaces are important:

- `AgentAdapter`: drives a native lmms agent or bridges an external OpenAI/Anthropic/Google/MCP-compatible scaffold.
- `Environment`: `reset`, `step`, `snapshot`, `restore`, and `verify`.
- `Trajectory`: immutable event history that can be replayed and independently scored.
- `Simulator`: a separately identified model/policy with its own failures and reliability metrics.

Report success, policy compliance, invalid action rate, tool efficiency, cost/latency, pass@k or pass^k, and simulator/environment error separately. Do not collapse them into one JSON string returned as a model response.

### P1.2: Generalize execution and resume

Define an executor protocol such as `submit`, `status`, `cancel`, `resume`, `collect`, and `detect`, with implementations for local subprocess, Accelerate/torchrun, Slurm, Kubernetes/Ray, and provider batch APIs. Executors should consume the same immutable plan and write the same event/result schema.

Generation, deterministic scoring, LLM judging, aggregation, and publication should be independent idempotent stages. This allows expensive GPU inference, cheap CPU scoring, and remote judging to use different resources without task-specific launch scripts.

### P1.3: Turn judging and statistics into decision evidence

Judge improvements:

- Pin judge model/deployment, prompt, parser, generation config, and adapter version.
- Repeat borderline grading, support multiple judges, randomize or swap pairwise order, and retain every individual judgment.
- Measure agreement against a human-labeled calibration set and between judges; report disagreement and infrastructure failures.
- Neutralize structural delimiters in model-controlled text and test the judge prompt against injection cases.

Statistical improvements:

- Declare the experimental unit and cluster for every metric.
- Use a test appropriate to the score type rather than a single paired t-test for all metrics.
- Report effect size and interval before p-value; correct across a benchmark suite.
- Make missing/failed/refused samples and repeated trials explicit in denominators.
- Add machine-readable quality gates with `GO`, `NO-GO`, and `INCONCLUSIVE`, never infer a quality verdict from job completion.

### P2: Governance, interoperability, and diagnosis

Once the contracts above are stable:

- Version plugin manifests and provide conformance fixtures for models, tasks, scorers, tools, sandboxes, executors, and exporters.
- Export/import Inspect-style logs or an agreed neutral schema instead of adding direct dependencies on every harness.
- Validate leaderboard submissions from run bundles, with immutable suite versions and eligibility tiers.
- Add transcript scanners for refusal, evaluation awareness, suspicious shortcuts, prompt injection, reward hacking, and environment misconfiguration.
- Add adaptive sample sources and power-aware stopping only after observation units and resume identity are correct.

## What not to prioritize first

- More benchmark count as the headline feature. Coverage is already a strength and increases the cost of weak contracts.
- More bespoke model classes without capability declarations and conformance tests.
- A new UI before there is one versioned log/result schema for the UI to consume.
- Per-task judge wrappers that bypass a common `JudgeSpec` and calibration workflow.
- Replacing the existing response cache. Its crash recovery and rank isolation are useful; extend its identity inputs and connect it to the run bundle.
- A universal cross-framework abstraction that tries to emulate every harness. Prefer a small typed core plus environment/export adapters.

## Suggested success measures

- At least 95% of invalid task/model/modality combinations rejected by `validate` before expensive initialization.
- 100% provenance completeness for leaderboard-eligible runs: immutable model, dataset/media, code, environment, prompt/scorer, judge, and execution identity.
- Zero primary-model calls required to re-run deterministic scoring or a new judge over a completed run.
- Interrupted static runs resume with no completed sample repeated; interrupted agent runs restore the last committed environment/agent checkpoint.
- Generic judge suites report agreement, order sensitivity, failure rate, and calibration accuracy, not only mean score.
- External executable tasks cannot access host filesystem, network, or unrelated secrets unless the plan grants that capability.
- Every published result validates against the same result schema used by local analysis, the HTTP service, and the leaderboard.

## Sources

[1] OpenAI Evals, building and versioning an eval: https://github.com/openai/evals/blob/main/docs/build-eval.md

[2] OpenAI Evals API, data-source schemas and run objects: https://platform.openai.com/docs/api-reference/evals

[3] OpenAI Graders API, typed and composite graders: https://platform.openai.com/docs/api-reference/graders

[4] Inspect AI tasks: https://inspect.aisi.org.uk/tasks.html

[5] Inspect AI solvers and typed task state: https://inspect.aisi.org.uk/solvers.html

[6] Inspect AI sandboxing: https://inspect.aisi.org.uk/sandboxing.html

[7] Inspect AI multimodal content and media authority: https://inspect.aisi.org.uk/multimodal.html

[8] Inspect AI resumable eval sets: https://inspect.aisi.org.uk/eval-sets.html

[9] Inspect AI versioned eval logs: https://inspect.aisi.org.uk/eval-logs.html

[10] Inspect AI model grading and reproducible graders: https://inspect.aisi.org.uk/model-graded.html

[11] Inspect AI scoring metrics and multi-judge reliability: https://inspect.aisi.org.uk/metrics.html

[12] Inspect AI extension entry points: https://inspect.aisi.org.uk/extensions.html

[13] Inspect AI agent checkpointing and transcript scanners: https://inspect.aisi.org.uk/checkpointing.html and https://inspect.aisi.org.uk/scanners.html

[14] EleutherAI lm-evaluation-harness task configuration: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md

[15] EleutherAI lm-evaluation-harness decontamination: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md

[16] HELM specification/state/controller architecture: https://github.com/stanford-crfm/helm/blob/main/docs/code.md

[17] HELM project and maintenance-mode notice: https://github.com/stanford-crfm/helm

[18] LightEval typed task configuration: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/lighteval_task.py

[19] LightEval result and detail formats: https://huggingface.co/docs/lighteval/en/saving-and-reading-results

[20] LightEval vLLM parallelism options: https://github.com/huggingface/lighteval/blob/main/docs/source/use-vllm-as-backend.mdx

[21] NeMo Evaluator architecture and executor protocol: https://docs.nvidia.com/nemo/evaluator/architecture

[22] NeMo Evaluator result format: https://docs.nvidia.com/nemo/evaluator/evaluation/result-format

[23] NeMo Evaluator paired run comparison: https://docs.nvidia.com/nemo/evaluator/nightly/tutorials/compare

[24] NeMo Evaluator project, environments, resume, sandboxes, and exporters: https://github.com/NVIDIA-NeMo/Evaluator

[25] MLPerf Client source and configuration/integrity contracts: https://github.com/mlcommons/mlperf_client

[26] MLPerf Client benchmark component tiers and quality qualification: https://mlcommons.org/benchmarks/client/

[27] VLMEvalKit benchmark and multimodal message development contract: https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md

[28] VLMEvalKit runtime, distributed inference, reuse, and judge artifacts: https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md

[29] VLMEvalKit model/dataset configuration system: https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/ConfigSystem.md

[30] SWE-bench containerized evaluation and prediction/result formats: https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md

[31] SWE-bench layered Docker caching: https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/docker_setup.md

[32] SWE-agent strict environment/deployment configuration: https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/environment/swe_env.py

[33] tau2-bench domains, policies, tools, simulators, text, and voice: https://github.com/sierra-research/tau2-bench

[34] tau2-bench trajectory re-evaluation and review commands: https://github.com/sierra-research/tau2-bench/blob/main/docs/cli-reference.md

[35] tau2-bench trajectory-aware evaluator: https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/evaluator/evaluator.py

[36] lmms-eval issue #1406, group-level post-processing: https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/1406

[37] lmms-eval issue #1259, reasoning/evaluation decoupling: https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/1259

[38] lmms-eval issue #908, judge throughput and stage separation: https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/908

# RFC: A Safe Modular Evaluation Pipeline

Decision: build a safe persistent evaluation lane around frozen, transport-neutral contracts. Do not rewrite the general evaluator.

lmms-eval continues to own task compilation, scoring, aggregation, and compatibility adapters. A LoadedRunner owns preprocessing, batching, inference, model lifecycle, and per-run reset. Every controlled job selects either the `formal` lane or the `persistent` lane. The system never changes lanes as a fallback.

The first delivery preserves the current command line, Python, HTTP, MCP, TUI, model registry, request, result-dictionary, and result-file surfaces through adapters. Borrowed runners remain deferred until one concrete training caller can provide an immutable ownership contract.

## Motivation

The current evaluator already contains the task and scoring behavior that makes lmms-eval useful. Replacing it would put benchmark compatibility at risk while solving a narrower problem: repeated evaluations need explicit identity, isolated state, durable artifacts, and an optional model-reuse lane.

The current HTTP path is safe in one important respect: it runs each job in a subprocess. It is inefficient for repeated evaluations because every job reloads the model, and it recovers results by scanning timestamped files rather than receiving a job-owned manifest. A persistent lane should retain process isolation while reusing only model-owned state.[1]

The current low-level evaluator is not a safe lifecycle seam for a borrowed model. It resets process-global random generators, accepts a caller-created model directly, and later calls `clean()` unconditionally. The base cleanup implementation deletes every `torch.nn.Module` attribute it finds. The new design therefore owns its loaded model instead of borrowing one.[2]

The repository also has correctness, recovery, and security defects on the path to persistence. Those defects must land as small fixes before the lifecycle refactor, so the new lane does not preserve them behind a deeper interface.

Open issue 1259 asks for reasoning workflows to be decoupled from benchmark evaluation. This design supplies a narrow execution seam for that need while leaving general workflow orchestration outside lmms-eval.[16]

## Scope

This RFC changes the orchestration seam around the existing evaluator.

- lmms-eval owns task discovery, task compilation, document selection, few-shot construction, filtering, reasoning normalization, `process_results`, aggregation, and legacy adaptation.
- LoadedRunner owns one loaded model runtime, document-bound preprocessing, batching, inference, cache-hook attachment, per-run state reset, and final model cleanup.
- EvaluationControl owns admission, job state, idempotency, cancellation, and telemetry coalescing.
- EvaluationExecutor owns execution in either the `formal` or `persistent` lane.
- HTTP, MCP, TUI, the command line, and Python functions are adapters. They do not own job state or execution policy.

## Current pipeline and evidence

The current core flow is:

```text
flat or subcommand CLI
  -> argument and YAML resolution
  -> model and task resolution
  -> task request construction as Instance objects
  -> request grouping by model method
  -> model inference
  -> filter application
  -> scoring and aggregation
  -> result dictionary and timestamped result files
```

`TaskManager`, task YAML, and task callables define benchmark behavior. `simple_evaluate` loads the model and tasks, while `evaluate` builds requests, calls model methods, filters outputs, scores documents, and aggregates metrics. `EvaluationTracker` writes aggregated and sample artifacts.[3]

The HTTP server and MCP surface use a sequential scheduler. The scheduler builds an argument vector, creates one subprocess per job, waits for it, then scans model directories and selects the latest timestamp. The TUI has a separate execution path.[1][4]

### Current facts verified at the RFC baseline

Source facts below were verified against commit `c58c56f5` on 2026-08-22.

- Full pytest collection finds 461 tests and stops on two collection errors. One test imports an unqualified `utils` module, and OCRBench v2 requires an unavailable `jieba` dependency. This is a local validation baseline, not a performance result.[5]
- Request-cache loading happens even when request caching is disabled. Saving replaces callable request arguments with `None` on the live `Instance` objects before serialization.[6]
- A fractional sample limit is resolved into an integer by assigning back to the shared `limit` variable. Later tasks therefore receive the first task's integer instead of resolving the original fraction against their own document count.[7]
- Standard `reasoning_tags` stripping runs after task filter ensembles. An extraction filter can therefore consume reasoning text before the scoring loop strips it. The separate `auto_strip_thinking` path works around this by inserting a filter at the front.[8]
- Deterministic responses are appended to the recovery audit before validity is checked. Audit replay checks the deterministic flag and key, but does not apply response validation, so an empty or malformed deterministic response rejected by SQLite storage can be restored after a crash.[9]
- Cache construction derives a raw model fingerprint from model arguments and stores that fingerprint in SQLite metadata. Hashing other records does not make this raw metadata secret-safe.[9]
- Aggregated-result saving mutates the result dictionary and writes directly to the final file. Sample saving mutates each sample, appends one line at a time to the final file, and can expose a partial file after interruption.[10]
- Direct model injection still receives the default process-global seed resets, and `evaluate` calls `lm.clean()` regardless of whether the evaluator created the model.[2]
- Model names currently exist in static simple and chat maps, a V2 manifest registry derived from those maps, a legacy decorator registry, and plugin loaders. A single entry-point exception stops the V2 entry-point loop at the outer catch.[11]
- The TUI allows every CORS origin while allowing credentials. It accepts free-form environment setup and environment text, interpolates request fields into a shell string, and passes that string to `create_subprocess_shell`.[12]

### Evidence classes

Current facts are source behavior and local test output at the baseline commit. They justify correctness and safety work, but they do not predict speedups.

No local performance microbenchmark is accepted by this RFC. The evaluator does not yet expose phase boundaries, and a one-off timing would mix task discovery, dataset cache state, preprocessing, model load, inference, and artifact writes. PR 3 establishes the first acceptable local benchmark record.

Historical measurements remain useful as regression warnings, not current baselines. Issue 698 reports ChartQA `relaxed_overall` of 0.7232 through the Python backend and 0.4980 through vLLM for the same named Qwen2-VL checkpoint. The report is unresolved and was not reproduced for this RFC, so it establishes a parity requirement rather than a present backend verdict.[13]

Issue 1126 records the long-tail generation problem that motivated dynamic independent HF workers. The existing `async_hf_model` backend is the resulting opt-in worker-pool path. This RFC preserves that backend and does not replace its internal scheduling without new measurements.[14]

All throughput changes in the conditional roadmap are hypotheses. A hypothesis becomes scheduled work only after the promotion gate in this RFC is met.

## Design invariants

- The public semantic contracts are `EvalSpecV1` and `EvalResultV1`.
- Both contracts are immutable after validation, reject unknown top-level fields, and use canonical JSON encodings for fingerprints.
- The normative schemas, canonicalization rules, identifier domains, null rules, revision representation, callable digests, and Instance projections are defined in Appendix A.
- An `extensions` object is allowed only for namespaced observational metadata. Extensions cannot change task, model, execution, scoring, identity, or artifact semantics.
- A semantic change requires a new contract version. A V1 reader never guesses how to interpret a newer version.
- The `formal` lane uses a fresh isolated subprocess for one job.
- The `persistent` lane uses a process-isolated, single-slot worker that may reuse only a matching model runtime.
- Lane selection is explicit in the frozen spec. Unsupported persistent execution fails as persistent execution and never retries as formal execution.
- The control plane never imports or owns a GPU model.
- A completed job always points to one validated, immutable artifact manifest. A file scan is never the completion protocol.
- A persistent worker accepts only one run at a time. No task state, response state, random state, metric state, or cache hook survives reset.
- Persistent eligibility is default-deny. A backend without a complete capability declaration fails persistent admission.
- Borrowed, caller-owned model objects are not part of either new lane.

## Frozen contracts and identity

### EvalSpecV1

`EvalSpecV1` is the only value that crosses the control-to-execution seam. Frontend parsing and resolution may use mutable implementation data, but admission freezes the final spec before it receives a job identifier.

| Field | Required semantics |
| --- | --- |
| `schema_version` | The integer `1`. |
| `intent` | The normalized caller choices: requested model selector, ordered task selectors, non-secret model arguments, secret references, document selection, few-shot settings, generation settings, scoring settings, seeds, and artifact policy. |
| `resolved` | The canonical model resolution and ordered task resolutions, including task profile identifiers, revision records, and secret-safe digests. |
| `execution` | Exactly one lane, `formal` or `persistent`, plus bounded runtime controls. The lane is never `auto`. |
| `identity` | `intent_id`, `resolved_eval_id`, and `spec_digest`, computed as defined in Appendix A. |
| `provenance` | The lmms-eval version, `semantic_core_id`, and source revision used to resolve the spec. |
| `extensions` | Optional observational metadata that is JSON-safe and cannot affect execution. |

`intent_id` hashes the canonical caller intent, including the requested lane and artifact policy. It excludes the idempotency key, `latest_for`, queue metadata, timestamps, and secret values.

`task_profile_id` hashes one task's evaluation meaning. It includes the canonical task name, resolved include closure, prompt and formatter definitions, dataset identity and split, document selection, few-shot policy, generation overrides, filters, reasoning policy, scoring functions, aggregations, and relevant callable digests. It excludes the model backend, batch size, device topology, lane, cache location, output root, retry, and worker identity.

`resolved_eval_id` hashes the resolved model identity, ordered task profile identifiers, model generation semantics, seeds, and `semantic_core_id`. An immutable release supplies a release artifact or source-manifest digest. An unreleased build supplies the exact source commit plus a semantic source-tree digest. The identifier excludes lane, hardware, batching, queue state, output location, and attempt. Formal and persistent runs can therefore be compared under one resolved evaluation identity while retaining distinct runtime identities.

`runtime_id` is not part of EvalSpec identity. The executor derives it from the lane, model runtime and library versions, device topology, batching controls, immutable load key, and worker generation. It never includes the attempt number. `run_id` identifies the admitted job, while `attempt_id` identifies one execution attempt. None of these identifiers is reused as a task profile.

Fingerprints never contain, log, persist, or hash raw secret values. Sensitive keys, URL credentials, tokens, passwords, and authorization headers become explicit secret references before canonicalization. A secret reference may include an operator-provided opaque generation label, but never the credential value. Inline secrets that cannot be converted to a reference are rejected at controlled remote admission. Local compatibility adapters may inject them only after identity construction and must redact every result, log, cache record, and manifest projection.

### EvalResultV1

`EvalResultV1` is an immutable, manifest-independent evaluation result. It contains evaluation identities, metrics, task records, telemetry, and provenance, but no manifest path, manifest digest, artifact entry, or publication state. Failed and cancelled jobs retain structured job state and diagnostic artifacts, but they do not fabricate a successful EvalResultV1. A telemetry job replaced by `latest_for` remains in the `cancelled` state with a structured supersession reason.

| Field | Required semantics |
| --- | --- |
| `schema_version` | The integer `1`. |
| `identity` | `intent_id`, `resolved_eval_id`, `spec_digest`, `runtime_id`, `run_id`, `attempt_id`, and the ordered task profile identifiers. |
| `metrics` | Canonical per-task metric and standard-error values, with filter names and higher-is-better metadata retained. |
| `groups` | Canonical group aggregations and hierarchy metadata. |
| `task_records` | Resolved task versions, non-secret configs, document counts, request counts, cache counts, and warnings. |
| `telemetry` | Pre-publication evaluation phase durations and counters. It contains no artifact path, manifest data, or publication timing. |
| `provenance` | Source revision, package versions relevant to execution, model revision, and dataset revisions when available. |
| `warnings` | Structured warning codes and messages. |
| `extensions` | Optional observational metadata with the same restrictions as EvalSpecV1. |

EvalResultV1 is complete when every selected task has a terminal task record and required metrics or an explicit metric omission reason. Raw generations remain in separate sample records. Scoring consumes normalized generations, while sample records retain both raw and normalized values when they differ.

After atomic publication, EvaluationControl constructs a separate `PublishedEvalResultRefV1` containing only `run_id`, `attempt_id`, the manifest-relative path, and the manifest digest. Controlled job state exposes it when requested legacy projections complete and the job becomes `completed`. The manifest hashes the serialized EvalResultV1 and every published artifact. EvalResultV1 never points back to that manifest.

The legacy result dictionary is a deterministic projection that consumes EvalSpecV1, EvalResultV1, and sample records where task hashes or sample files require them. Required sanitized versions and resolved configs come from task records. The adapter preserves existing keys, nesting, and serializable values and never infers reconstructable values from task, config, callable, or result digests.

## Instance remains the request record

`Instance` gains named, document-bound views for context, generation settings, document identity, task and split identity, message construction, media construction, target construction, and the bound document. New core and runner code uses these views instead of tuple positions.

The named views do not eagerly decode media. They bind task semantics to a document during task compilation, then let the LoadedRunner perform model-specific preprocessing at execution time.

`.args` remains a read-only, versioned compatibility adapter over the V1 named views. The existing tuple layouts for `loglikelihood`, configurable `generate_until`, message-based `generate_until`, multi-round generation, and existing agentic generation stay covered by contract tests. A layout change requires a new adapter version. New code cannot add another positional interpretation.

The design does not add a second `PreparedRequest` type. Internal ephemeral batch objects are implementation details inside LoadedRunner and cannot cross the runner seam. A second public request type is justified only if two independently maintained runner adapters need different stable request representations.

Appendix A defines the required named fields and exact `.args` projections for current main. Open PR 1418's proposed `generate_until_game` request shape is unmerged and is not frozen by this RFC. Slice 5 starts only after PR 1418 lands or explicitly rebases; it must preserve the landed game contract through the same Instance seam and cannot introduce a competing agentic request type.

## Target flow

```text
CLI, Python, HTTP, MCP, or TUI input
  -> compatibility adapter
  -> core resolution and frozen EvalSpecV1
  -> EvaluationControl admission
  -> selected EvaluationExecutor
       formal: one fresh subprocess
       persistent: one process-isolated single-slot worker
  -> core task compilation
  -> LoadedRunner preprocessing, batching, and inference
  -> core filtering, normalization, scoring, and aggregation
  -> EvalResultV1, sample records, and staged job-owned artifacts
  -> atomic bundle publication and manifest validation
  -> PublishedEvalResultRefV1 in controlled job state
  -> terminal job state
  -> legacy result and filename adapters
```

Task compilation resolves task definitions, includes, datasets, selected document indices, few-shot context, and document-bound Instances. It does not load the model or decode all media into model inputs.

LoadedRunner accepts one EvalSpecV1 transaction at a time. It invokes the existing task and scoring behavior through internal seams, but it owns every resource that can remain live across runs.

## LoadedRunner lifecycle

A LoadedRunner owns the model, processor, tokenizer, backend clients, device placement, backend caches, batch planner, and model-specific preprocessing state. The runner has one external capability: execute one frozen spec atomically against a matching runtime and return result material plus telemetry.

A persistent worker may reuse its LoadedRunner only when the resolved model identity and immutable runtime-loading controls match. A different identity retires the worker and starts a new persistent worker. Unsupported topology or backend behavior fails admission or execution in the persistent lane.

Persistent eligibility is declared by the selected backend and defaults to denied. The declaration contains exactly `capability_id`, `immutable_load_key_fields`, `reset_hook`, `health_check`, `close_hook`, `poison_hook`, `supported_topologies`, `local_judge_mode`, and `validation_suite`. `local_judge_mode` is `co_resident`, `after_model_close`, or `unsupported`; every other field is non-empty and the topology field is a non-empty ordered array. Missing, partial, unknown, or unvalidated declarations fail persistent admission. No adapter is inferred safe from its base type, and no failure falls back to the formal lane.

Before each run, the runner establishes the specified Python, NumPy, Torch, and few-shot random state and clears request responses, filters, token counters, metric accumulators, usage budgets, task context, cache hooks, logging context, and model task dictionaries.

After a successful run, the runner detaches run-owned task and cache state, resets all per-run state, runs a lightweight health check, and only then reports itself reusable. Final model cleanup occurs when the worker retires, not after every successful persistent run.

Once a runner has begun a job, an uncaught exception, cancellation, timeout, worker communication failure, invalid response cardinality, out-of-memory failure, reset failure, or failed health check poisons the worker. The executor fails the current attempt, terminates the worker process, and creates a new worker only for a later attempt or job. It does not retry the current job or change lanes unless an explicit retry policy creates a new attempt.

Validation errors completed before the runner accepts the job do not poison an existing worker because they cannot mutate it.

## EvaluationControl and EvaluationExecutor

EvaluationControl owns the monotonic job states `queued`, `running`, `completed`, `failed`, and `cancelled`. Completion requires a validated manifest. Cancellation never reports completion, and a running job cannot return to queued.

Admission validates the frozen spec, lane support, configured resource bounds, output-root policy, idempotency metadata, and telemetry coalescing metadata. Control records state and events, but delegates all model and evaluator work to EvaluationExecutor.

An idempotency key is scoped to the configured caller namespace. Repeating a key with the same complete non-secret spec digest returns the existing job in any retained state. Repeating the key with a different digest returns a conflict. A failed or cancelled job is not rerun under the same key. Idempotency records live for the advertised job-retention interval.

`latest_for` is accepted only for jobs marked as telemetry. After idempotency resolution, admission of newer telemetry cancels older jobs with the same caller namespace and `latest_for` value only while those jobs are queued. A running job continues. Completed and failed history remains. Each replaced job stays `cancelled` and records `reason.kind="superseded"` plus the new `run_id`.

EvaluationExecutor receives an admitted job and its EvalSpecV1, starts the selected lane, emits structured phase and lifecycle events, and returns either a PublishedEvalResultRefV1 after validated publication or a structured failure. It does not own queue policy or idempotency.

The formal executor always creates a fresh child process and passes structured data without shell parsing. The persistent executor communicates with one isolated worker process and schedules one job at a time. Worker replacement stays within the persistent lane and is not a fallback to formal execution.

HTTP and MCP translate their request and response schemas to EvaluationControl. TUI uses the same control interface for execution. An MCP wait mode such as current `auto`, `async`, or `sync` controls client waiting only and is distinct from the execution lane.

## Artifact contract

Every admitted job owns one final bundle under a configured output root. The final path is `{output_root}/runs/{run_id}/attempts/{attempt_id}` after each opaque identifier passes path-segment validation. It is not derived from a model name, task name, timestamp scan, or caller path fragment.

The executor writes into a unique staging directory under the same filesystem as the final bundle. It serializes EvalResultV1 as an ordinary payload beside per-task samples, logs, and telemetry. The manifest lists and hashes the serialized result and every artifact by logical role, relative path, media type, byte size, and digest.

Publication has one linearization sequence. The publisher closes and fsyncs every payload file, writes and fsyncs the manifest last in staging, fsyncs the staging directory, renames the complete directory to the exact `run_id` plus `attempt_id` final path without overwrite, then fsyncs the final parent directory. A digest-valid final manifest after the parent fsync is authoritative.

If a worker dies after the rename but before EvaluationControl records the terminal state, startup recovery and repeated admission use the recorded `run_id` and `attempt_id` to inspect that exact bundle. A valid matching manifest resumes any requested legacy projection, creates PublishedEvalResultRefV1 from the verified final path and digest, and reconciles the job to `completed`. Recovery never selects a timestamp or a latest directory.

An existing final bundle with the same spec digest, run identifier, attempt identifier, manifest digest, and payload digests is an idempotent publication. An existing bundle with mismatched identity is an `artifact_conflict`; an invalid manifest or payload digest is `artifact_corruption`. Control marks the attempt failed, preserves the evidence, and never overwrites, deletes, repairs, or adopts the conflicting bundle automatically. A staging directory is never authoritative.

The legacy adapter consumes EvalSpecV1, EvalResultV1, and the sample records listed in the verified canonical bundle. It projects canonical bytes into `{output_path}/{sanitized_model_name}/{timestamp}_samples_{task_name}.jsonl` and `{output_path}/{sanitized_model_name}/{timestamp}_results.json`. It atomically writes and fsyncs every sample alias first, then atomically writes and fsyncs the aggregate result alias last as the projection commit marker, followed by a parent-directory fsync. Existing identical aliases are idempotent; an existing alias with different bytes fails projection without overwrite.

Job completion requires every legacy projection requested by the caller. A retryable projection failure leaves the canonical bundle valid and the job `running` while bounded recovery retries projection from that exact bundle. A projection conflict or exhausted retry policy moves the attempt to `failed`; a failed attempt never later becomes completed. The HTTP v1 response continues to expose its current model-to-result-path mapping, but those paths come from the manifest and completed projection rather than a latest-file scan.

Legacy path projection is enabled by default for EvalSpecV1 and remains supported throughout the V1 contract lifetime. Removing the default requires a new major contract, migration documentation, two released deprecation cycles, and evidence that maintained consumers use manifest roles. Canonical bundle retention follows the configured control policy; projected files follow the caller-owned `output_path` retention policy and are never garbage-collected by EvaluationControl.

Artifact serialization never mutates EvalResultV1, task records, or sample records. Each output is derived from an immutable snapshot.

## Performance metric contract

PR 3 lands measurement before EvalSpecV1, resolved identities, runtime identities, and canonical bundle publication exist. It emits a standalone `BaselinePerformanceRecordV1` with source commit and tree digest, normalized legacy arguments or their safe digest, environment, hardware, cache state, repetition identity, phase intervals, counters, and resources. It contains no EvalSpec, resolved, runtime, run, or attempt identifier.

PR 10 can translate the same instrumentation into canonical `PerformanceRecordV1` after EvalSpec and identities exist. PR 12 lists canonical performance records in the manifest and may retain a BaselinePerformanceRecordV1 as a separately hashed evidence input. The baseline and canonical records are different schemas and are never claimed to have identical bytes.

Every measured run records the same stable phases where applicable:

- `queue_wait` covers admission to executor start and is measured by EvaluationControl.
- `model_load` covers runtime construction and is zero only for a verified persistent reuse.
- `task_resolution` covers task lookup, includes, and dataset metadata resolution.
- `request_build` covers document selection, few-shot construction, and Instance creation.
- `preprocess` covers document-bound message and media conversion into backend inputs.
- `inference` covers backend submission through complete raw responses.
- `filter_and_normalize` covers filters and reasoning removal.
- `score` covers task result processing.
- `aggregate` covers metric and group aggregation.
- `artifact_stage` covers serializing EvalResultV1 and other payloads before the performance record and manifest are written.
- `reset` covers per-run cleanup and the reuse health check.

The manifest fsync, directory rename, and parent-directory fsync are recorded as out-of-band control lifecycle timing after publication. They cannot be embedded in EvalResultV1 or a file inside the bundle without creating a self-reference.

Durations use monotonic clocks within the process that owns the phase. Cross-process wall times are reported separately and are not summed from clocks with different origins. If phases overlap, each phase records its own interval and the report states that the phase sum can exceed end-to-end wall time.

Every benchmark records selected documents, built Instances, responses, batches, cache hits and misses, raw and normalized output counts, input and output tokens when available, peak host RSS, peak GPU allocation when available, model-load reuse, and failure counts. Throughput reports both end-to-end scored documents per second excluding queue wait and inference tokens per second when token counts exist.

### Representative workloads

- A hermetic CPU dummy-model run covers two tasks with different sizes, fractional limits, request caching on and off, scoring, aggregation, and artifacts.
- A local HF vision-language run covers variable image sizes and long-tail output lengths, including `async_hf_model` as its own backend rather than a replacement runner policy.
- A Qwen3-VL video run separates media decode and processor preparation from GPU inference.
- Deterministic vLLM and SGLang runs use the same pinned checkpoint, task profiles, documents, prompts, generation settings, and seeds as a local Python backend parity run.
- An API-backed judge run reports request concurrency, rate-limit waits, retries, tokens, latency, and cost without exposing credentials.
- Formal cold-start and persistent cold-first-run plus warm-reuse runs use the same resolved evaluation identity.

### Benchmark methodology and promotion rule

Each report pins the source commit, source-tree digest, model and dataset revisions when available, container or environment lock, hardware, backend versions, normalized legacy arguments or EvalSpecV1 according to record type, and cache state. Before PR 10, BaselinePerformanceRecordV1 is the evidence file. After PR 10, canonical runs use PerformanceRecordV1. After PR 12, the manifest references the applicable record as a hashed artifact.

Deterministic comparisons use identical selected document identifiers and compare raw outputs, normalized outputs, per-sample scores, aggregate metrics, and artifact counts before comparing speed. Issue 698 makes output and score parity a release gate, not an optional validation.[13]

Each configuration runs one unreported warm-up when meaningful and at least five measured repetitions. Reports include median and p95 end-to-end time, phase durations, throughput, peak memory, and variance. Persistent reports separate model-load avoidance from steady-state execution.

An optimization is promotable only when its conditional gate is met, correctness parity passes, and the representative workload shows at least a 10 percent median improvement in the named bottleneck or a documented reliability improvement. A single local timing cannot promote work.

## Compatibility and migration policy

| Surface | V1 policy |
| --- | --- |
| Flat and subcommand CLI | Preserve both `lmms-eval --model ...` and `lmms-eval eval --model ...`. Both normalize to the same semantic fields. Existing invocations default through the adapter to `formal`; new core callers must provide a lane. |
| Python facades | Preserve `simple_evaluate`, `evaluate`, arguments, synchronous return behavior, and result dictionaries. Serializable high-level calls can normalize to EvalSpecV1. Direct model and task objects remain on the quarantined legacy in-process adapter and cannot request persistent execution. |
| HTTP v1 | Preserve routes, request and response shapes, five job states, and queued-only cancellation semantics. Omitted lane maps explicitly to `formal`. Additive V1 fields may request `persistent`, idempotency, and telemetry coalescing. Sync and async wait helpers must stop on every terminal state: return completed and cancelled job records, and raise for failed jobs. The current cancelled-job polling bug is not compatibility behavior.[4] |
| MCP and TUI | Preserve tool names, wait behavior, previews, and visible job semantics. Execution routes through EvaluationControl. TUI previews may render a command, but execution never evaluates that string. |
| Registries and plugins | Make ModelRegistryV2 authoritative. Preserve names, aliases, legacy maps as compatibility views, the legacy plugin environment hook, and entry points. Load and report each plugin independently so one failure cannot hide healthy plugins. |
| `Instance.args` | Preserve current tuple layouts through the V1 adapter and contract tests. New internals use named views. |
| Result dictionaries and files | Preserve current dictionary keys and timestamped aggregated and per-task sample filename forms through adapters. Projection consumes EvalSpecV1, manifest-independent EvalResultV1, and sample records. Controlled job state exposes PublishedEvalResultRefV1 separately. |

Compatibility adapters are deliberate public surfaces, not temporary bridges. This roadmap does not remove the flat CLI, Python functions, HTTP v1, `.args`, registry names, result dictionary, or legacy filenames.

### Temporary bridges and removal gates

| Bridge | Exact removal gate |
| --- | --- |
| EvalSpecV1 calling through the legacy evaluator argument bundle | Remove after both executors consume EvalSpecV1 directly, every existing output type passes adapter-parity tests, and PR 13's lifecycle suite is required. |
| EvalResultV1 wrapping a completed legacy result dictionary | Remove after scoring and aggregation construct EvalResultV1 directly, the legacy dictionary is generated only as a projection, and golden snapshots match for all contract tasks. |
| HTTP completion using the timestamp-file scanner | Remove after every successful job publishes a validated manifest, HTTP and MCP consume manifest roles, and crash tests prove no partial bundle can complete a job. |
| Scheduler forwarding shim between current routes and EvaluationControl | Remove after HTTP, MCP, and TUI share control-state tests and no adapter imports the legacy scheduler state store. |
| Mutable static model maps feeding ModelRegistryV2 | Stop internal writes after registry parity and plugin fault-isolation tests pass. Keep read-only compatibility views until a future major version has migration documentation and one full release of deprecation evidence. |

## Security and threat model

The HTTP, MCP, and TUI servers remain trusted-network tools. This RFC does not make them safe for direct public or hostile multi-tenant exposure. Deployment still requires network isolation, and public exposure requires separate authentication, authorization, rate limiting, and tenant isolation work.[4]

Trusted callers can still make mistakes or submit malformed values. The controlled paths therefore defend against shell injection, path traversal, symlink escape, accidental secret persistence, duplicate execution, unbounded queue growth, and corrupted partial artifacts.

- Executors use argument arrays or structured IPC. No request field, environment value, setup text, model argument, task name, or path is interpolated into a shell command.
- Remote adapters do not accept arbitrary shell setup. Environment configuration uses an operator allowlist and secret references.
- Controlled output paths resolve under one configured root. Validation rejects absolute child paths, parent traversal, symlink escape, existing final bundles, and cross-job writes.
- Logs, errors, specs, results, manifests, cache metadata, fingerprints, command previews, and telemetry pass through one secret-redaction policy.
- Admission bounds queue length, spec size, task count, artifact policy, and lane-specific resources.
- Task callables, model adapters, datasets, and plugins remain trusted executable code. A persistent worker is a fault-isolation boundary, not a sandbox.

## Required roadmap

The required program contains exactly 16 delivery slices. Each row is independently reviewable and keeps legacy parity unless its title names a correctness or security fix. A physical implementation PR can cover only one slice.

| PR | Exact title | Depends on | Exit gate |
| ---: | --- | --- | --- |
| 1 | `docs(rfc): freeze evaluation contracts, identities, lanes, and compatibility policy` | None | This RFC is accepted with the contracts and scope frozen. |
| 2 | `ci: make the hermetic CPU contract suite green and required` | 1 | Full hermetic collection and the contract subset pass in required CI, including the two current collection failures. |
| 3 | `perf: add phase timings and a reproducible benchmark harness` | 2 | The representative CPU workload produces standalone BaselinePerformanceRecordV1 JSON without future contract or job identities. |
| 4 | `fix(evaluator): resolve sample limits independently per task` | 2 | Mixed-size multi-task fractional-limit tests prove independent document counts in build and scoring. |
| 5 | `fix(requests): add named Instance views and versioned tuple projections` | 2, 4, PR 1418 coordination | Current request types match Appendix A; `.args` is derived and read-only; the landed PR 1418 game contract is preserved without another agentic seam. |
| 6 | `fix(scoring): strip reasoning before extraction and retain raw generations` | 2 | Extraction sees normalized text, sample artifacts retain raw and normalized generations, and existing opt-in behavior remains compatible. |
| 7 | `fix(cache): make request and response caching fail closed and secret-safe` | 2, 5 | Disabled request caching performs no read; replacement serialization never mutates Instances; response recovery rejects invalid data; merge leases fail closed; cache files contain no raw secrets. |
| 8 | `security(tui): restrict origins and remove interpolated eval shell commands` | 2 | Origin tests, injection tests, and structured subprocess execution pass. |
| 9 | `refactor(models): make ModelRegistryV2 authoritative with fault-isolated plugins` | 2 | All current names and aliases resolve identically, legacy views match, and one broken plugin does not hide another. |
| 10 | `feat(core): normalize CLI and Python into immutable EvalSpec and identities` | 3, 4, 5, 6, 7, 9 | Flat CLI, subcommand CLI, and serializable Python fixtures produce matching Appendix A specs, `semantic_core_id`, secret-safe identities, and canonical PerformanceRecordV1. |
| 11 | `feat(core): introduce EvalResult with legacy dict and result-file adapters` | 10 | Manifest-independent EvalResultV1 carries exact legacy version and config sources; golden projections from EvalSpec, EvalResult, and samples match current dictionaries and files. |
| 12 | `fix(artifacts): atomically publish immutable job-owned run bundles` | 3, 11 | Fault injection proves publication and recovery return PublishedEvalResultRefV1; manifests hash result and artifacts; baseline evidence and canonical performance records remain distinct. |
| 13 | `refactor(runner): make LoadedRunner own model lifecycle and run EvalSpec atomically` | 3, 5, 7, 10, 11, 12 | One-shot runner parity passes, reset invariants pass, and failure poisoning terminates owned model state. |
| 14 | `refactor(server): separate EvaluationControl from SubprocessExecutor` | 8, 10, 12, 13 | HTTP v1, MCP, and TUI normalize into the same EvalSpec; five-state and terminal-wait tests pass; formal execution remains subprocess-isolated. |
| 15 | `feat(server): add an explicit process-isolated single-slot persistent lane` | 3, 13, 14 | Sequential reuse avoids a second model load, parity matches formal, unsupported requests fail without fallback, and poison replacement is proven. |
| 16 | `feat(control): add idempotency and queued-only telemetry coalescing` | 14, 15 | Duplicate-key, conflict, retention, state-race, and queued-only `latest_for` tests pass through HTTP and MCP adapters. |

The 16 rows are required delivery slices, not permission to grow a review. For code delivery, any physical PR expected to exceed approximately 400 changed lines, 8 files, or one coherent concern must split into smaller PRs within its row. A split cannot absorb work from another row, weaken the row's exit gate, or reorder dependencies. The documentation-only RFC can exceed the line guideline because it remains one file and one contract concern.

### Required PR waves and DAG

| Wave | PRs | Rule |
| --- | --- | --- |
| Contract baseline | 1, 2 | No implementation lands before the RFC and hermetic gate. |
| Measure and repair | 3 through 9 | These slices may proceed in parallel only where the dependency column permits. Slice 7 owns both request-cache replacement and fail-closed response-cache safety. |
| Freeze core values | 10 through 12 | EvalSpec precedes EvalResult, and EvalResult precedes canonical artifact publication. |
| Establish ownership | 13, 14 | Runner ownership lands before the server delegates execution through the new seam. |
| Add reuse and policy | 15, 16 | Persistent reuse lands before control coalescing is enabled for that lane. |

Open PR 1418 adds an agentic game runner and currently overlaps `api/instance.py`, `api/task.py`, `evaluator.py`, and `api/registry.py`. Slice 5 starts only after PR 1418 lands or explicitly rebases and coordinates with it; later overlapping evaluator work follows the same rule. The modular pipeline must preserve its landed game request contract, reuse its accepted agentic seams, and must not create a competing agentic request or runner hierarchy.[15]

## Conditional roadmap

The following 12 bets are not scheduled by default. Meeting a gate permits a proposal and benchmark, not automatic implementation.

| Bet | Conditional work | Promotion gate |
| ---: | --- | --- |
| 1 | atomically materialize one proven checkpoint format | One named formal or persistent caller repeatedly pays a measured materialization cost or suffers partial-checkpoint failures, and the format has an immutable source-to-output identity. |
| 2 | borrowed-runner lane for an identified immutable training caller | One named training integration can prove exclusive model ownership during eval, immutable weights and processor state, explicit random-state policy, cancellation semantics, and a lifecycle test that the training owner accepts. |
| 3 | extract task preparation only with measured or maintainability evidence | `task_resolution` plus `request_build` exceeds 15 percent of end-to-end time on two representative workloads, or three accepted changes require the same preparation edit across three current modules. |
| 4 | extract execution/cache only with measured or maintainability evidence | Cache and execution plumbing exceeds 10 percent of measured wall time, causes a reproduced recovery defect after PR 7, or three accepted changes duplicate the same seam. |
| 5 | extract scoring/aggregation only with measured or maintainability evidence | `score` plus `aggregate` exceeds 15 percent on two workloads, or three accepted benchmark changes require coordinated edits across the same scoring path. |
| 6 | shard document indices before row materialization | Request-build peak RSS exceeds the agreed worker budget or row materialization exceeds 15 percent of end-to-end time on a production-size task, with identical selected document identifiers after sharding. |
| 7 | batch response-cache commits and shard merges transactionally | Cache commit and merge time exceeds 10 percent of end-to-end time or lease telemetry shows repeated contention, and crash injection proves the batched transaction cannot admit invalid recovery rows. |
| 8 | overlap Qwen3-VL CPU preparation with GPU generation | Qwen3-VL `preprocess` exceeds 15 percent while GPU idle time exceeds 15 percent on the same run, and bounded overlap preserves request order, raw outputs, scores, and peak-memory limits. |
| 9 | gather one ordered distributed payload per task | Distributed gather and reconstruction exceeds 10 percent of task wall time or creates a reproduced ordering defect, and the single payload stays within the measured host-memory budget. |
| 10 | vLLM bounded batching/concurrency | A pinned vLLM workload shows at least 10 percent headroom after phase instrumentation, and raw-output plus score parity with the reference backend passes within explicit memory and queue bounds. |
| 11 | SGLang bounded batching/concurrency | A pinned SGLang workload shows at least 10 percent headroom after phase instrumentation, and raw-output plus score parity passes within explicit memory and queue bounds. |
| 12 | opt-in bounded judge batching/streaming | Judge time exceeds 30 percent of end-to-end time, the provider contract supports the proposed mode, rate and cost limits are explicit, and per-sample judge inputs, outputs, and scores remain auditable. |

## Validation matrix

| Concern | Required evidence |
| --- | --- |
| Schema stability | Golden RFC 8785 bytes, absent-versus-null cases, unknown nested-field rejection, revision unions, callable CRLF, lone-CR, missing-final-newline, and multiple-final-newline digest cases, V1 round trips, mutation rejection, and V2 rejection tests. |
| Identity | Equivalent frontend fixtures produce equal intent and resolved identities; same-version semantic source changes alter `semantic_core_id` and `resolved_eval_id`; runtime changes alter runtime identity only; attempt changes do not; secrets never appear. |
| Core correctness | Mixed task sizes, every Appendix A Instance projection, raw-versus-normalized reasoning, cache hit and recovery, scoring, aggregation, and result snapshots pass on CPU. |
| Formal lane | Subprocess isolation, cancellation process-group cleanup, structured argv, manifest completion, failure state, and legacy HTTP behavior pass. |
| Persistent lane | Default-deny capability tests, cold run, at least 20 sequential warm runs, load-key mismatch rotation, reset and health invariants, local-judge modes, supported topologies, cancellation, timeout, OOM simulation, poison termination, and no cross-run contamination pass. |
| Artifact durability | Kill injection before and after rename, file and directory fsync evidence, exact-attempt PublishedEvalResultRefV1 reconciliation, conflict and corruption handling, sample-first legacy projection, aggregate-last commit marker, and alias parity pass. EvalResult fixtures remain byte-identical before and after publication. |
| Frontend parity | Flat CLI, subcommand CLI, and serializable Python generate equivalent specs in slice 10; HTTP v1, MCP, and TUI join in slice 14; every wait helper terminates on completed, failed, and cancelled states. |
| Registry compatibility | Every built-in name and alias resolves as before; legacy views match V2; plugin failures are isolated and reported. |
| Security | Origin allowlist, injection corpus, output traversal and symlink tests, queue bounds, spec-size bounds, and secret scans across logs, cache, results, telemetry, and manifests pass. |
| Legacy reconstruction | Projection tests rebuild `versions`, `configs`, aggregate results, task hashes, and sample files from EvalSpecV1, EvalResultV1, and sample records without reverse-engineering a digest. |
| Performance | Slice 3 emits BaselinePerformanceRecordV1 with no future identities; slice 10 emits canonical PerformanceRecordV1; slice 12 may hash both as distinct artifacts; five-run parity, memory, and variance satisfy the benchmark contract. |
| Distributed behavior | Ordered document and response identity, rank failure, barrier failure, cache merge, and one result manifest pass on each supported executor topology. |
| Agentic overlap | The accepted PR 1418 request and runner tests pass without a duplicate agentic seam after each overlapping core change. |

## Non-goals

- This RFC does not rewrite the evaluator or split every current phase into a public module.
- This RFC does not introduce `PreparedRequest`, a new task language, or a second scoring framework.
- This RFC does not change benchmark definitions or aggregate metric semantics except for the named correctness fixes.
- This RFC does not replace `async_hf_model`, vLLM, SGLang, or model-specific batching internals without a promoted bet.
- This RFC does not add parallel slots to the persistent worker.
- This RFC does not add a borrowed-runner or mutable training-model path.
- This RFC does not generalize checkpoint conversion before one format meets its gate.
- This RFC does not sandbox untrusted task code, model code, datasets, or plugins.
- This RFC does not expose the servers to untrusted networks.
- This RFC does not remove legacy public surfaces in the 16 required PRs.
- This RFC does not add silent retries, lane fallback, or automatic performance tuning.
- This RFC does not duplicate the agentic runner seams under review in PR 1418.

## Risks and mitigations

- A frozen contract can preserve accidental behavior. V1 freezes semantic inputs and named views, while legacy tuple and file accidents remain adapters rather than the canonical model.
- Secret redaction can collapse distinct credentials. Stable operator-owned secret references and optional opaque generation labels preserve intent without persisting values.
- Persistent model state can contaminate later runs. Single-slot execution, exhaustive reset tests, poison-on-uncertainty, and process replacement make reuse fail closed.
- Atomic directory publication varies across filesystems. Staging and final directories must share one filesystem, and unsupported output roots fail admission.
- Adapter drift can make frontends disagree. Golden spec and result fixtures run through every adapter in required CI.
- Faster backends can change answers. Raw-output and score parity precede throughput claims, with issue 698 retained as the concrete warning.
- The roadmap can grow into a general refactor. Conditional extraction work remains unscheduled until its explicit gate is met.
- PR 1418 can conflict with early request work. Overlapping PRs coordinate after its merge or rebase, and its agentic seam remains authoritative.

## Appendix A: Normative V1 contracts

Appendix A is normative. If explanatory prose elsewhere conflicts with this appendix, Appendix A wins for EvalSpecV1, EvalResultV1, PublishedEvalResultRefV1, BaselinePerformanceRecordV1, PerformanceRecordV1, identifiers, revisions, callable digests, and Instance projections.

### Type, presence, and extension rules

The words MUST, MUST NOT, REQUIRED, OPTIONAL, and MAY have their usual normative meanings.

`string` means valid Unicode encoded as UTF-8 for serialization, with no lone surrogate. Classification is by mathematical value, not source-language or lexical type. `integer` means any JSON number whose mathematical value is integral and lies in the inclusive IEEE 754 safe-integer range `-9007199254740991` through `9007199254740991`. V1 rejects integral values outside that range. `number` means a finite, mathematically non-integral IEEE 754 binary64 value. NaN, positive or negative infinity, and negative zero are forbidden. A later schema version MAY define an explicit tagged decimal-integer subtype. V1 never guesses or coerces a string into an integer. `json_value` means null, boolean, integer, number, string, an array of `json_value`, or an object with string keys and `json_value` values.

A REQUIRED field is always present and never null unless its row explicitly includes null. An OPTIONAL field is omitted when unavailable. Structural null is forbidden. Absence and null are distinct: absence means the contract field does not apply or was not supplied, while null is data and is permitted only inside the explicitly open maps `intent.model.arguments`, `intent.generation.arguments`, `extensions`, `metrics`, `groups`, performance counters and resources, and `task_records[*].sanitized_version`.

Every object defined by a field table has `additionalProperties=false`. Fields explicitly typed as maps permit arbitrary keys whose values have only the stated map value type. The open `json_value` maps named in the previous paragraph permit arbitrary JSON values. `extensions` keys MUST contain one owner namespace separator `/`, and extensions are observational only. A reader MUST reject an extension that claims to change execution or evaluation semantics.

Arrays preserve order. Empty arrays and empty objects are values, not aliases for absence. Duplicate object keys are invalid. Set-like inputs are sorted before they enter a contract; ordered task selectors and task resolutions are never sorted.

### Revisions and digests

`RevisionRefV1` is exactly one of the following objects:

```json
{"kind":"immutable","value":"non-empty revision string"}
{"kind":"unavailable"}
```

The `value` field is REQUIRED for `kind="immutable"` and forbidden for `kind="unavailable"`. Empty strings, null, `unknown`, and mutable branch names do not represent unavailable revisions. An adapter uses `kind="unavailable"` only after it attempted the resolver appropriate to that source.

Contract identifiers use SHA-256 and the lowercase external form `sha256:<64 lowercase hexadecimal characters>`. Given an ASCII domain string and a payload, the digest input is `UTF8(domain)`, one zero byte, then the canonical JSON bytes of the payload. The V1 domains are exact:

| Identifier | Domain |
| --- | --- |
| `intent_id` | `lmms-eval/EvalSpecV1/intent-id` |
| `task_profile_id` | `lmms-eval/EvalSpecV1/task-profile-id` |
| `resolved_eval_id` | `lmms-eval/EvalSpecV1/resolved-eval-id` |
| `spec_digest` | `lmms-eval/EvalSpecV1/spec-digest` |
| `runtime_id` | `lmms-eval/EvalResultV1/runtime-id` |
| callable digest | `lmms-eval/CallableDigestV1` |
| safe model-arguments digest | `lmms-eval/EvalSpecV1/safe-model-arguments` |
| resolved task-config digest | `lmms-eval/EvalSpecV1/task-config` |
| semantic core manifest digest | `lmms-eval/SemanticCoreV1/manifest` |
| baseline safe legacy-arguments digest | `lmms-eval/BaselinePerformanceRecordV1/safe-legacy-arguments` |
| source-tree digest | `lmms-eval/SourceTreeV1/manifest` |

Canonical JSON follows RFC 8785 JSON Canonicalization Scheme. Serialization is UTF-8 without a byte-order mark or trailing newline. Object keys use the RFC 8785 order, arrays retain input order, strings receive no Unicode normalization, and numbers use RFC 8785 serialization after the V1 numeric restrictions above. File digests in an artifact manifest are plain SHA-256 over exact file bytes and use the same external form without a domain prefix.

`intent_id` hashes `{"intent": intent, "lane": execution.lane}`. `task_profile_id` hashes the task-profile projection defined below. `resolved_eval_id` hashes `{"resolved_model": resolved.model, "task_profile_ids": [resolved.tasks[*].task_profile_id], "generation": intent.generation, "scoring": intent.scoring, "seeds": intent.seeds, "semantic_core_id": provenance.semantic_core_id}`. `spec_digest` hashes every non-secret field in `schema_version`, `intent`, `resolved`, `execution`, and `provenance`; it excludes `identity` to avoid recursion and excludes observational `extensions`. Secret values never enter a contract projection.

`runtime_id` hashes the exact RuntimeDescriptorV1 fields `lane`, `backend_id`, `backend_version`, `immutable_load_key`, `device_topology`, `batching`, `worker_generation`, and `local_judge_mode`. `attempt_id` is never part of RuntimeDescriptorV1 or `runtime_id`.

### semantic_core_id

`semantic_core_id` identifies code that can change task compilation, request meaning, filtering, normalization, scoring, or aggregation even when the package version is unchanged.

An immutable release MUST use `release:{package_version}:sha256:{digest}`. The digest covers the release's build-embedded semantic source manifest. If an older immutable release lacks that manifest, the resolver deterministically hashes the installed semantic files and records that derived manifest with the result.

An unreleased build MUST use `source:{full_source_commit}:sha256:{digest}` for a clean tree or `source:{full_source_commit}+dirty:sha256:{digest}` for a modified tree. The digest covers the semantic source manifest plus the exact tracked diff and any untracked semantic files. The full source commit is REQUIRED even for a dirty build. A controlled run whose unreleased source commit cannot be determined fails resolution instead of substituting a package version or `unavailable`.

`provenance.semantic_core_id` is REQUIRED in EvalSpecV1 and EvalResultV1, and the exact string is part of `resolved_eval_id`. Two installations with the same package version but different semantic code therefore cannot share a resolved evaluation identity.

### CallableDigestV1

Every callable that affects a task profile MUST have a CallableDigestV1 value. A callable can provide an explicit immutable semantic digest through task or plugin metadata. Without one, resolution uses the following deterministic projection and never falls back to `repr`, an address, object identity, or pickle bytes.

- A Python function projection contains `kind="python"`, module, qualified name, `source_sha256`, canonical defaults, canonical keyword defaults, and canonical closure-cell values. It never contains raw source bytes or source text.
- A bound method uses the underlying function projection plus the declaring class module and qualified name. Receiver identity is excluded because resolved task configuration is hashed separately.
- A `functools.partial` projection contains `kind="partial"`, the recursively computed base callable digest, canonical positional arguments, and canonical keyword arguments.
- A callable without readable source uses `kind="package-symbol"`, module, qualified name, Python implementation and version, and the immutable containing distribution or file digest.
- A decorated callable hashes the callable actually invoked. Automatic unwrapping is forbidden unless an explicit semantic digest declares the wrapper equivalent.
- Defaults, closure cells, partial values, or receiver configuration that affect behavior but cannot be represented canonically require an explicit immutable semantic digest. If neither deterministic projection nor explicit digest is available, controlled resolution fails.

To compute `source_sha256`, decode the retrieved source as Unicode, replace every CRLF and remaining CR with LF, remove all trailing LF characters, append exactly one LF, encode as UTF-8, and compute plain SHA-256 over those bytes. The JSON field is the lowercase string `sha256:{64_hex}`. Invalid Unicode or unavailable source uses the package-symbol rule instead of inserting bytes into JSON.

The callable digest is the V1 domain-separated SHA-256 of its JSON projection. The task-profile projection contains a map from stable semantic role names, such as `doc_to_messages` or `process_results`, to these digests. Source paths, line numbers, raw source text, and raw source bytes are not part of the projection.

### EvalSpecV1 schema

The exact top-level EvalSpecV1 fields are:

| Path | Type | Presence and meaning |
| --- | --- | --- |
| `schema_version` | integer literal `1` | REQUIRED. |
| `intent` | IntentV1 | REQUIRED. |
| `resolved` | ResolvedV1 | REQUIRED. |
| `execution` | ExecutionV1 | REQUIRED. |
| `identity` | SpecIdentityV1 | REQUIRED. |
| `provenance` | SpecProvenanceV1 | REQUIRED. |
| `extensions` | object of namespaced `json_value` | OPTIONAL; excluded from identifiers. |

IntentV1 has the following exact fields:

| Path | Type | Presence and meaning |
| --- | --- | --- |
| `intent.model.selector` | non-empty string | REQUIRED; caller spelling before canonical model resolution. |
| `intent.model.arguments` | object of `json_value` | REQUIRED; default `{}`; secret-bearing argument slots are absent and appear in `secret_refs`. |
| `intent.model.secret_refs` | object of non-empty string values | REQUIRED; default `{}`; keys match the argument slots they supply. |
| `intent.model.force_simple` | boolean | REQUIRED. |
| `intent.task_selectors` | non-empty array of non-empty strings | REQUIRED; caller order is preserved. |
| `intent.selection.limit` | integer `-1`, integer at least `1`, or number strictly between `0` and `1` | OPTIONAL; absence means the full selected split. |
| `intent.selection.offset` | integer at least `0` | REQUIRED. |
| `intent.selection.repeats` | integer at least `1` | REQUIRED. |
| `intent.selection.num_fewshot` | integer at least `0` | OPTIONAL; absence retains each task's resolved default. |
| `intent.prompting.system_instruction` | string | OPTIONAL; empty string is allowed and distinct from absence. |
| `intent.prompting.apply_chat_template` | boolean | REQUIRED. |
| `intent.prompting.fewshot_as_multiturn` | boolean | REQUIRED. |
| `intent.generation.arguments` | object of `json_value` | REQUIRED; default `{}`; task-specific merged values remain in task profiles. |
| `intent.scoring.bootstrap_iters` | integer at least `0` | REQUIRED. |
| `intent.scoring.predict_only` | boolean | REQUIRED. |
| `intent.scoring.reasoning_tags` | array of two-string arrays | OPTIONAL; absence selects task or core defaults; empty array explicitly disables stripping. |
| `intent.scoring.process_with_media` | boolean | REQUIRED. |
| `intent.seeds.python` | integer or string literal `unmanaged` | REQUIRED. |
| `intent.seeds.numpy` | integer or string literal `unmanaged` | REQUIRED. |
| `intent.seeds.torch` | integer or string literal `unmanaged` | REQUIRED. |
| `intent.seeds.fewshot` | integer or string literal `unmanaged` | REQUIRED. |
| `intent.artifacts.log_samples` | boolean | REQUIRED. |
| `intent.artifacts.write_out` | boolean | REQUIRED. |
| `intent.artifacts.legacy_projection` | boolean | REQUIRED. |

ResolvedV1 has the following exact fields:

| Path | Type | Presence and meaning |
| --- | --- | --- |
| `resolved.model.canonical_id` | non-empty string | REQUIRED. |
| `resolved.model.adapter_kind` | string enum `simple` or `chat` | REQUIRED. |
| `resolved.model.adapter_id` | non-empty opaque string | REQUIRED; stable registry identity, not a file path. |
| `resolved.model.adapter_revision` | RevisionRefV1 | REQUIRED. |
| `resolved.model.model_revision` | RevisionRefV1 | REQUIRED. |
| `resolved.model.safe_arguments_digest` | digest string | REQUIRED; computed after secret replacement. |
| `resolved.model.persistent_capability_id` | non-empty string | OPTIONAL; absence means persistent admission is denied. |
| `resolved.tasks` | non-empty array of ResolvedTaskV1 | REQUIRED; expanded task order is preserved. |
| `resolved.tasks[*].requested_selector` | non-empty string | REQUIRED. |
| `resolved.tasks[*].canonical_name` | non-empty string | REQUIRED. |
| `resolved.tasks[*].task_profile_id` | digest string | REQUIRED. |
| `resolved.tasks[*].dataset_revision` | RevisionRefV1 | REQUIRED. |
| `resolved.tasks[*].split` | non-empty string | REQUIRED. |
| `resolved.tasks[*].config_digest` | digest string | REQUIRED. |
| `resolved.tasks[*].callable_digests` | object of digest-string values | REQUIRED; default `{}` only when no callable affects semantics. |

The task-profile projection is exactly `canonical_name`, resolved include closure and config, dataset source and RevisionRefV1, split, selection values applicable to that task, few-shot policy, prompting policy, merged generation arguments, filter and reasoning policy, scoring and aggregation configuration, and `callable_digests`. It excludes model, lane, runtime, cache paths, output paths, and attempt data.

ExecutionV1, SpecIdentityV1, and SpecProvenanceV1 have the following exact fields:

| Path | Type | Presence and meaning |
| --- | --- | --- |
| `execution.lane` | string enum `formal` or `persistent` | REQUIRED. |
| `execution.batch_size` | integer at least `1` or non-empty adapter-supported string | REQUIRED. |
| `execution.max_batch_size` | integer at least `1` | OPTIONAL. |
| `execution.device` | non-empty string | OPTIONAL. |
| `execution.num_gpus` | integer at least `1` | REQUIRED. |
| `execution.distributed_backend` | string enum `accelerate` or `torchrun` | REQUIRED. |
| `execution.response_cache` | non-empty string | OPTIONAL. |
| `execution.request_cache_mode` | string enum `off`, `read_write`, `refresh`, or `delete` | REQUIRED. |
| `execution.output_root` | non-empty string | REQUIRED; controlled adapters resolve and confine it before execution. |
| `execution.resource_profile` | non-empty string | OPTIONAL. |
| `identity.intent_id` | digest string | REQUIRED. |
| `identity.resolved_eval_id` | digest string | REQUIRED. |
| `identity.spec_digest` | digest string | REQUIRED. |
| `provenance.lmms_eval_version` | non-empty string | REQUIRED. |
| `provenance.semantic_core_id` | semantic core string defined above | REQUIRED. |
| `provenance.source_revision` | RevisionRefV1 | REQUIRED. |

### EvalResultV1 schema

The exact top-level EvalResultV1 fields are:

| Path | Type | Presence and meaning |
| --- | --- | --- |
| `schema_version` | integer literal `1` | REQUIRED. |
| `identity` | ResultIdentityV1 | REQUIRED. |
| `metrics` | object of `json_value` | REQUIRED; default `{}` only for predict-only runs. |
| `groups` | object of `json_value` | REQUIRED; default `{}`. |
| `task_records` | non-empty array of TaskResultV1 | REQUIRED. |
| `telemetry` | EvaluationTelemetryV1 | REQUIRED. |
| `provenance` | ResultProvenanceV1 | REQUIRED. |
| `warnings` | array of WarningV1 | REQUIRED; default `[]`. |
| `extensions` | object of namespaced `json_value` | OPTIONAL; excluded from identifiers. |

ResultIdentityV1 and task records have the following exact fields:

| Path | Type | Presence and meaning |
| --- | --- | --- |
| `identity.intent_id` | digest string | REQUIRED; copied from the spec. |
| `identity.resolved_eval_id` | digest string | REQUIRED; copied from the spec. |
| `identity.spec_digest` | digest string | REQUIRED; copied from the spec. |
| `identity.runtime_id` | digest string | REQUIRED; attempt number excluded. |
| `identity.run_id` | non-empty opaque string | REQUIRED. |
| `identity.attempt_id` | non-empty opaque string | REQUIRED. |
| `identity.task_profile_ids` | non-empty array of digest strings | REQUIRED; same order as resolved tasks. |
| `task_records[*].task_name` | non-empty string | REQUIRED. |
| `task_records[*].task_profile_id` | digest string | REQUIRED. |
| `task_records[*].sanitized_version` | `json_value` | REQUIRED; null is allowed only here to preserve a legacy null version. |
| `task_records[*].sanitized_config` | object of `json_value` | REQUIRED; the full resolved JSON-safe task config, not a digest. |
| `task_records[*].document_count` | integer at least `0` | REQUIRED. |
| `task_records[*].request_count` | integer at least `0` | REQUIRED. |
| `task_records[*].metric_keys` | array of strings | REQUIRED; default `[]`. |
| `task_records[*].metric_omission_reason` | non-empty string | REQUIRED only when an expected metric is absent; otherwise forbidden. |
| `task_records[*].cache.hits` | integer at least `0` | REQUIRED. |
| `task_records[*].cache.misses` | integer at least `0` | REQUIRED. |
| `task_records[*].cache.skipped` | integer at least `0` | REQUIRED. |
| `task_records[*].warnings` | array of WarningV1 | REQUIRED; default `[]`. |

WarningV1 contains exactly `code` as a non-empty string and `message` as a non-empty string, plus OPTIONAL `task_name` as a non-empty string. ResultProvenanceV1 contains exactly REQUIRED `lmms_eval_version`, `semantic_core_id`, `source_revision`, `model_revision`, and `dataset_revisions`, where the first two are non-empty strings, the revisions use RevisionRefV1, and `dataset_revisions` maps canonical task names to RevisionRefV1.

EvaluationTelemetryV1 contains exactly REQUIRED `phases`, `counters`, and `resources`. Its phase entries use PhaseRecordV1 but may contain only `queue_wait`, `model_load`, `task_resolution`, `request_build`, `preprocess`, `inference`, `filter_and_normalize`, `score`, `aggregate`, and `reset`. It excludes `artifact_stage` and every publication timing so EvalResultV1 bytes are final before staging begins.

EvalResultV1 has no `publication`, `manifest`, `manifest_path`, `manifest_digest`, `entries`, or artifact-reference field. Controlled job state exposes PublishedEvalResultRefV1 out of band only after publication. PublishedEvalResultRefV1 contains exactly REQUIRED `run_id`, `attempt_id`, `manifest_path`, and `manifest_digest`; the first two are non-empty opaque strings, `manifest_path` is a confined non-empty relative path, and `manifest_digest` is a file-byte digest. Extra fields are forbidden.

The legacy projection reads `task_records[*].sanitized_version` to reconstruct `versions` and `task_records[*].sanitized_config` to reconstruct `configs`. It consumes EvalSpecV1 for run-level configuration and naming, EvalResultV1 for metrics and task records, and sample records for samples and task hashes. It MUST NOT derive any of these values by reversing or substituting `task_profile_id`, `config_digest`, callable digests, or manifest digests.

BaselinePerformanceRecordV1 contains exactly REQUIRED `schema_version=1`, `record_kind="baseline"`, `source_commit`, `source_tree_digest`, `legacy_invocation`, `environment_lock_digest`, `hardware`, `cache_state`, `repetition`, `phases`, `counters`, and `resources`. `source_commit` is a full non-empty commit string, digests use the V1 digest form, `hardware` is a non-empty string, and `cache_state` is `cold`, `warm`, `mixed`, or `disabled`.

`legacy_invocation` is a discriminated union. The normalized form contains exactly `kind="normalized"` and `arguments` as a secret-safe canonical legacy argument map. The digest form contains exactly `kind="digest"` and `safe_digest` as a V1 digest string. `repetition` contains exactly REQUIRED non-empty `suite_id`, non-empty `case_id`, integer `repetition_index` at least zero, and boolean `warmup`. BaselinePerformanceRecordV1 forbids EvalSpec, intent, resolved, runtime, run, and attempt identifiers. PR 3 emits this pre-contract record.

PerformanceRecordV1 contains exactly REQUIRED `schema_version=1`, `record_kind="canonical"`, `run_id`, `attempt_id`, `resolved_eval_id`, `runtime_id`, `phases`, `counters`, `resources`, and `benchmark_context`. `benchmark_context` contains exactly REQUIRED `source_commit`, `source_tree_digest`, `environment_lock_digest`, `hardware`, `cache_state`, `warmup_count`, and `measured_repetitions`; types and cache values match the baseline schema, and counts are non-negative integers with at least one measured repetition. PR 10 can emit this record after canonical identities exist.

Both record types use the same PhaseRecordV1: `phases` is an array whose entries contain exactly `name`, `owner`, `duration_ns`, and `overlapped`; `name` uses the stable phase names in this RFC, `owner` is `control`, `worker`, or `publisher`, `duration_ns` is an integer at least zero, and `overlapped` is boolean. `counters` and `resources` are open maps of `json_value`. PR 12 hashes whichever records it publishes as separate manifest artifacts; retaining a baseline record as evidence does not convert it into or assert byte equality with PerformanceRecordV1.

### InstanceV1 named views and `.args` projection

InstanceV1 is an internal request contract, not a transport object. Every real and padding request has REQUIRED non-null named views `request_type`, `context`, `document`, `doc_id`, `task_name`, `split`, `repeats`, `idx`, and `padding_only`. `context` is a string and may be empty. `document` is the bound non-null dataset value. `doc_id` and `idx` are integers at least zero, `repeats` is an integer at least one, `task_name` and `split` are non-empty strings, and `padding_only` is boolean.

Type-specific named views are either REQUIRED and non-null in the matrix below or absent. They are never null. `generation_arguments` is an immutable mapping. `visual_builder`, `messages_builder`, `text_builder`, and `target_builder` are callables whose semantic digests live in the task profile. `continuation` is a string and may be empty.

| Current request form | REQUIRED type-specific named views | Exact `.args` V1 projection |
| --- | --- | --- |
| Simple `generate_until` and `generate_visual_cot` | `generation_arguments`, `visual_builder` | `(context, generation_arguments, visual_builder, doc_id, task_name, split)` |
| Chat `generate_until` | `messages_builder`, `generation_arguments` | `(context, messages_builder, generation_arguments, doc_id, task_name, split)` |
| Direct task `loglikelihood` | `target_builder`, `visual_builder` | `(context, target_builder, visual_builder, doc_id, task_name, split)` |
| Multiple-choice `loglikelihood` | `continuation`, `visual_builder` | `(context, continuation, visual_builder, doc_id, task_name, split)` |
| Mutual-information unconditional `loglikelihood` compatibility form | `continuation` | `(context, continuation)` |
| Simple `generate_until_multi_round` | `generation_arguments`, `visual_builder`, `text_builder` | `(context, generation_arguments, visual_builder, text_builder, doc_id, task_name, split)` |
| Chat `generate_until_multi_round` | `messages_builder`, `generation_arguments` | `(context, messages_builder, generation_arguments, doc_id, task_name, split)` |
| Current simple or chat `generate_until_agentic` | `generation_arguments`, `visual_builder`, `text_builder` | `(context, generation_arguments, visual_builder, text_builder, doc_id, task_name, split)` |

`.args` derives a fresh tuple from named views and is read-only as an attribute. The projection preserves callable objects and per-Instance generation mapping semantics expected by current backends. New code reads named views. A new or changed projection requires a contract-version decision and parity tests across every registered consumer.

Open PR 1418's `generate_until_game` shape is not part of this matrix because it is not on current main. Slice 5 cannot start until PR 1418 lands or rebases against the slice. After that coordination, slice 5 adds the landed shape to the matrix and preserves it through InstanceV1 rather than adding a parallel request record or agentic runner seam.

## References

[1] Current HTTP subprocess and latest-file behavior: [`lmms_eval/entrypoints/job_scheduler.py`](../../lmms_eval/entrypoints/job_scheduler.py#L301-L367) and [`lmms_eval/entrypoints/job_scheduler.py`](../../lmms_eval/entrypoints/job_scheduler.py#L520-L560).

[2] Current model injection, seed, and cleanup behavior: [`lmms_eval/evaluator.py`](../../lmms_eval/evaluator.py#L243-L398), [`lmms_eval/evaluator.py`](../../lmms_eval/evaluator.py#L1107-L1109), and [`lmms_eval/api/model.py`](../../lmms_eval/api/model.py#L166-L176).

[3] Current core pipeline anchors: [`lmms_eval/cli/dispatch.py`](../../lmms_eval/cli/dispatch.py#L94-L146), [`lmms_eval/evaluator.py`](../../lmms_eval/evaluator.py#L830-L925), [`lmms_eval/api/task.py`](../../lmms_eval/api/task.py#L428-L580), and [`lmms_eval/loggers/evaluation_tracker.py`](../../lmms_eval/loggers/evaluation_tracker.py#L169-L212).

[4] Current trusted-network warning, HTTP adapters, and terminal-wait gap: [`lmms_eval/entrypoints/http_server.py`](../../lmms_eval/entrypoints/http_server.py#L30-L36), [`lmms_eval/entrypoints/http_server.py`](../../lmms_eval/entrypoints/http_server.py#L97-L138), and [`lmms_eval/entrypoints/client.py`](../../lmms_eval/entrypoints/client.py#L31-L58).

[5] Local validation on 2026-08-22: `PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest --collect-only -q -p no:cacheprovider` returned 461 collected tests and two collection errors; the errors were `test/eval/qwen2_5_vl/test_qwen2_5_vl.py` importing `utils` and `test/eval/test_ocrbench_v2.py` missing `jieba`.

[6] Current mutating request cache: [`lmms_eval/caching/cache.py`](../../lmms_eval/caching/cache.py#L25-L61) and unconditional cache load in [`lmms_eval/api/task.py`](../../lmms_eval/api/task.py#L453-L480).

[7] Current shared limit mutation: [`lmms_eval/evaluator.py`](../../lmms_eval/evaluator.py#L953-L999) and sample-size resolution in [`lmms_eval/evaluator_utils.py`](../../lmms_eval/evaluator_utils.py#L288-L295).

[8] Current filter and reasoning order: [`lmms_eval/evaluator.py`](../../lmms_eval/evaluator.py#L1123-L1149) and [`lmms_eval/evaluator.py`](../../lmms_eval/evaluator.py#L1181-L1235).

[9] Current response-cache fingerprint and recovery order: [`lmms_eval/caching/response_cache.py`](../../lmms_eval/caching/response_cache.py#L426-L480), [`lmms_eval/caching/response_cache.py`](../../lmms_eval/caching/response_cache.py#L548-L631), and [`lmms_eval/caching/response_cache.py`](../../lmms_eval/caching/response_cache.py#L779-L817).

[10] Current non-atomic and mutating artifact writes: [`lmms_eval/loggers/evaluation_tracker.py`](../../lmms_eval/loggers/evaluation_tracker.py#L185-L212) and [`lmms_eval/loggers/evaluation_tracker.py`](../../lmms_eval/loggers/evaluation_tracker.py#L252-L300).

[11] Current registry surfaces and plugin loading: [`lmms_eval/models/__init__.py`](../../lmms_eval/models/__init__.py#L18-L164), [`lmms_eval/models/__init__.py`](../../lmms_eval/models/__init__.py#L177-L248), [`lmms_eval/models/registry_v2.py`](../../lmms_eval/models/registry_v2.py#L139-L159), and [`lmms_eval/api/registry.py`](../../lmms_eval/api/registry.py#L8-L32).

[12] Current TUI origin and shell behavior: [`lmms_eval/tui/server.py`](../../lmms_eval/tui/server.py#L35-L46), [`lmms_eval/tui/server.py`](../../lmms_eval/tui/server.py#L288-L318), and [`lmms_eval/tui/server.py`](../../lmms_eval/tui/server.py#L543-L569), [`lmms_eval/tui/server.py`](../../lmms_eval/tui/server.py#L645-L680).

[13] Historical backend parity report: [GitHub issue 698, Performance gap between Python and vLLM](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/698).

[14] Historical async HF motivation and current registration: [GitHub issue 1126, Async evaluation for HF models](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/1126), [`lmms_eval/models/__init__.py`](../../lmms_eval/models/__init__.py#L148-L162), and [`lmms_eval/models/chat/async_hf_model.py`](../../lmms_eval/models/chat/async_hf_model.py#L36-L80).

[15] Current implementation overlap: [GitHub PR 1418, agentic game-loop evaluation](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1418).

[16] Current demand for a reasoning/evaluation seam: [GitHub issue 1259, Decouple reasoning and evaluation](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/1259).

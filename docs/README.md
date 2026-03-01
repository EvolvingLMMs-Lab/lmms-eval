# LMMs-Eval Documentation

This documentation covers every layer of `lmms-eval` — from running your first evaluation to adding custom models and tasks. The framework evaluates large multimodal models across image, video, and audio benchmarks with a single unified pipeline.

## Table of Contents

- [How the Evaluation Pipeline Works](#how-the-evaluation-pipeline-works)
- [Why it's Efficient and Trustworthy](#why-its-efficient-and-trustworthy)
  - [Efficient](#efficient)
  - [Trustworthy](#trustworthy)
- [Getting Started](#getting-started)
- [Extending the Framework](#extending-the-framework)
  - [Adding a Model](#adding-a-model)
  - [Adding a Task](#adding-a-task)
- [Using lmms-eval as a Library](#using-lmms-eval-as-a-library)
- [Performance and Caching](#performance-and-caching)
- [Task Catalog](#task-catalog)
- [Release Notes](#release-notes)
- [Additional Resources](#additional-resources)

## How the Evaluation Pipeline Works

Every evaluation follows the same six-stage pipeline. Each stage has dedicated documentation, and failures at any stage produce clear error messages indicating what went wrong.

```
User input: --model openai --tasks mmmu_val,video_mmmu,longvideobench_val_v
         │
         ▼
    ┌─ CLI Parsing ─────────────── commands.md
    │   Parse flags, resolve config, dispatch to evaluator
    │
    ├─ Model Resolution ────────── model_guide.md
    │   Map model name -> Python class, validate chat/simple type
    │
    ├─ Task Loading ────────────── task_guide.md
    │   Discover YAML configs, build prompts, load datasets
    │
    ├─ Model Inference ─────────── run_examples.md
    │   Send prompts to model, collect responses
    │
    ├─ Response Caching ────────── caching.md
    │   Store deterministic responses, skip redundant calls
    │
    └─ Metric Aggregation ──────── throughput_metrics.md
        Score responses, compute task-level metrics, write output
```

A minimal evaluation runs the entire pipeline with a single command. This example evaluates GPT-4.1-mini on MMMU via the OpenAI-compatible API:

```bash
export OPENAI_API_KEY="your-api-key"

python -m lmms_eval \
  --model openai \
  --model_args model_version=gpt-4.1-mini \
  --tasks mmmu_val \
  --batch_size 1 \
  --limit 8
```

The same pattern works for any OpenAI-compatible endpoint (OpenRouter, Azure, local vLLM/SGLang servers). To evaluate across image and video tasks together:

```bash
python -m lmms_eval \
  --model openai \
  --model_args model_version=gpt-4.1-mini \
  --tasks mmmu_val,video_mmmu,longvideobench_val_v \
  --batch_size 1 \
  --limit 8 \
  --log_samples \
  --output_path ./results/
```

## Why it's Efficient and Trustworthy

The pipeline above is designed to be fast enough to iterate on and rigorous enough to trust. This section summarizes the concrete mechanisms that back those claims. Each links to deeper documentation.

### Efficient

Evaluation should not be the bottleneck. Four layers of optimization keep GPUs and API endpoints saturated end to end.

| Layer | Mechanism | Impact |
|-------|-----------|--------|
| **API throughput** | Adaptive concurrency control with refill scheduling, prefix-aware queueing, and retry/backoff decoupling. | ~7.5x throughput over v0.5 on fixed benchmarks ([v0.6 release notes](releases/lmms-eval-0.6.md)). |
| **Response caching** | SQLite + JSONL write-ahead log stores deterministic responses (`temperature=0`, `do_sample=False`). Subsequent runs skip inference entirely for cached samples. | Zero redundant model calls on repeated or resumed runs ([caching guide](advanced/caching.md)). |
| **Video I/O** | TorchCodec multi-threaded decode replaces single-threaded PyAV. Lance-backed blob storage on Hugging Face enables single-IOP random access per video. | Up to 3.58x faster frame decode; eliminates full-table scans ([v0.7 release notes](releases/lmms-eval-0.7.md)). |
| **Prefix KV reuse** | Requests are clustered by shared media and sorted by length so vLLM/SGLang can maximize KV cache hits across a batch. | Fewer redundant prefill computations on shared-image/video tasks. |

The async pipeline decouples model inference from metric scoring. Model outputs are persisted immediately, and scoring runs independently - so a crash after inference does not lose results, and judge-stage work (exact match, LLM-as-judge) can run on a separate machine or at a later time.

### Trustworthy

A benchmark score is only useful if it is reproducible and statistically grounded. The framework provides four categories of trust guarantees.

**Reproducibility.** Every run is seeded (`--seed` controls Python `random`, NumPy, and PyTorch) and every task config is fingerprinted. The cache key includes a SHA-256 of the schema version, task YAML, generation kwargs, and document ID - so any change to the prompt, parameters, or task definition automatically invalidates stale results. The [test suite](../test/README.md) includes prompt stability snapshots for 8 classic benchmarks to catch unintended prompt regressions.

**Statistical rigor.** Point estimates hide uncertainty. The framework reports confidence intervals and supports clustered standard errors for benchmarks with correlated questions (e.g., multiple questions per video). Paired comparison with a baseline model removes question-difficulty variance, isolating the actual model difference with a p-value - replacing hand-waving with verifiable claims. Power analysis tools help determine the minimum sample size needed to detect a given improvement. See the [v0.6 release notes](releases/lmms-eval-0.6.md) for the full statistical methodology.

**Cache integrity.** Responses are validated before storage: `None`, empty strings, and malformed loglikelihood tuples are rejected. The JSONL write-ahead log is fsynced before the SQLite upsert, so a crash between the two writes is recovered on the next startup. Per-rank files prevent write contention in distributed runs. Details in the [caching guide](advanced/caching.md).

**Clean scoring.** Reasoning models (Qwen3-VL, DeepSeek-R1, QwQ) emit `<think>...</think>` blocks that must not leak into metric computation. The pipeline strips reasoning tags before scoring and preserves the raw output in a separate `resps` field for analysis. This is configured globally via `--reasoning_tags` or per-task in the YAML config. See the [commands guide](getting-started/commands.md) for usage.

## Getting Started

Start here if this is your first time using `lmms-eval`.

| Guide | What You Learn |
|-------|---------------|
| [Quick Start](getting-started/quickstart.md) | Clone, install, and run your first evaluation in 5 minutes. |
| [Commands Guide](getting-started/commands.md) | Every CLI flag explained — model selection, task filtering, batching, caching, output control, and seed management. |
| [Run Examples](getting-started/run_examples.md) | Copy-paste commands for LLaVA, Qwen, InternVL, VILA, GPT-4o, and other models across image, video, and audio tasks. |

## Extending the Framework

These guides walk through adding your own models and tasks.

### Adding a Model

The [Model Guide](guides/model_guide.md) covers the full process: subclass `lmms_eval.api.model.lmms`, implement `generate_until`, and register a `ModelManifest`. Chat models (recommended) receive structured messages with roles and typed content. Simple models (legacy) receive plain text with `<image>` placeholders.

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms

@register_model("my_model")
class MyModel(lmms):
    is_simple = False  # chat model

    def generate_until(self, requests):
        for request in requests:
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            messages = doc_to_messages(self.task_dict[task][split][doc_id])
            # ... run inference and store result ...
```

### Adding a Task

The [Task Guide](guides/task_guide.md) explains the YAML configuration format. Each task defines a dataset source, prompt template, generation parameters, and scoring function. The simplest tasks require only a YAML file; complex tasks add a `utils.py` with custom prompt formatting and metric computation.

```yaml
task: "my_benchmark"
dataset_path: "my-org/my-dataset"
test_split: test
output_type: generate_until
doc_to_messages: !function utils.my_doc_to_messages
process_results: !function utils.my_process_results
generation_kwargs:
  max_new_tokens: 1024
  temperature: 0
metric_list:
  - metric: acc
```

## Using lmms-eval as a Library

The [External Usage](advanced/external_usage.md) guide covers three access patterns beyond the standard CLI:

**CLI task browsing** lists registered tasks, groups, and model backends without downloading datasets:

```bash
lmms-eval tasks subtasks     # table of all leaf tasks with YAML paths
lmms-eval models --aliases   # all model backends with alias mappings
```

**Web UI** provides a browser-based interface for configuring and launching evaluations:

```bash
uv run lmms-eval-ui          # opens browser, requires Node.js 18+
```

**Python API** gives programmatic access to tasks, datasets, and the evaluator:

```python
from lmms_eval import evaluator

results = evaluator.simple_evaluate(
    model="openai",
    model_args="model_version=gpt-4.1-mini",
    tasks=["mmmu_val", "video_mmmu", "longvideobench_val_v"],
    batch_size=1,
    limit=8,
)
```

## Performance and Caching

| Guide | What You Learn |
|-------|---------------|
| [Caching](advanced/caching.md) | SQLite-backed response cache for deterministic requests. Store, replay, merge shards across distributed ranks, and recover from crashes via JSONL audit log. |
| [Throughput Metrics](advanced/throughput_metrics.md) | Inference timing metrics logged by chat models — end-to-end latency, time to first token, tokens per second, and batch-level summaries. |

The response cache stores only deterministic requests (`temperature=0`, `do_sample=False`). Enable it with `--use_cache ./eval_cache` to skip redundant model calls on repeated runs:

```bash
python -m lmms_eval \
  --model openai \
  --model_args model_version=gpt-4.1-mini \
  --tasks mmmu_val,video_mmmu \
  --use_cache ./eval_cache
```

## Task Catalog

The [Current Tasks](advanced/current_tasks.md) page lists every registered evaluation task across all modalities. The framework ships with 100+ tasks across five categories:

| Category | Benchmark | Task Name | What It Tests |
|----------|-----------|-----------|---------------|
| **Image** | MMMU | `mmmu_val` | College-level multimodal reasoning across 30 subjects. |
| | MME | `mme` | Perception and cognition across 14 subtasks. |
| | MathVista | `mathvista_testmini` | Mathematical reasoning with visual context. |
| | MMBench | `mmbench_en` | Multi-ability benchmark covering 20 fine-grained skills. |
| **Video** | Video-MMMU | `video_mmmu` | Knowledge acquisition from multi-discipline professional videos. |
| | VideoMME | `videomme` | Comprehensive video understanding across diverse content types. |
| | EgoSchema | `egoschema` | Long-form egocentric video reasoning. |
| **Long Video** | LongVideoBench | `longvideobench_val_v` | Extended video content with temporal reasoning. |
| **Audio** | AIR-Bench | `air_bench_chat` | Audio understanding across speech, music, and sound. |
| | LibriSpeech | `librispeech` | Automatic speech recognition accuracy. |
| **Agentic** | Vending-Bench 2 | `vending_bench2` | Multi-step tool-use with deterministic vending simulator (`generate_until_agentic`). |
| | τ2-Bench | `tau2_bench_telecom` | Telecom-domain tool-use with state tracking and `<tool_call>`/`<submit>` protocol. |

## Release Notes

Each release note documents new tasks, models, architectural changes, and migration steps.

| Version | Theme | Highlights |
|---------|-------|------------|
| [v0.7](releases/lmms-eval-0.7.md) | Operational simplicity | YAML-first config, reasoning-tag stripping, Lance-backed video, skill-based agent workflows. |
| [v0.6](releases/lmms-eval-0.6.md) | Evaluation as a service | Async HTTP server, adaptive API concurrency (~7.5x throughput), statistical rigor (CI, paired t-test). |
| [v0.5](releases/lmms-eval-0.5.md) | Audio expansion | Comprehensive audio evaluation, response caching, 50+ benchmark variants. |
| [v0.4](releases/lmms-eval-0.4.md) | Scale and reasoning | Distributed evaluation, reasoning benchmarks, unified chat interface. |
| [v0.3](releases/lmms-eval-0.3.md) | Audio foundations | Initial audio model support (Qwen2-Audio, Gemini-Audio). |

## Additional Resources

- [Dataset formatting tools](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/tools) — Scripts for converting datasets into lmms-eval compatible formats.
- [Test suite documentation](../test/README.md) — 292 tests covering every pipeline layer, with prompt stability snapshots for 8 classic benchmarks.
- [GitHub repository](https://github.com/EvolvingLMMs-Lab/lmms-eval) — Source code, issue tracker, and contributing guidelines.
- [Discord community](https://discord.gg/zdkwKUqrPy) — Get help and discuss development.

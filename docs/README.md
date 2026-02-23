# LMMs-Eval Documentation

This documentation covers every layer of `lmms-eval` — from running your first evaluation to adding custom models and tasks. The framework evaluates large multimodal models across image, video, and audio benchmarks with a single unified pipeline.

## How the Evaluation Pipeline Works

Every evaluation follows the same six-stage pipeline. Each stage has dedicated documentation, and failures at any stage produce clear error messages indicating what went wrong.

```
User input: --model openai_compatible --tasks mmmu_val,video_mmmu,longvideobench_val_v
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

* **[Commands Guide](commands.md)** - Learn about command line flags and options
* **[Quick Start](quickstart.md)** - Evaluate your model in 5 minutes
* **[Model Guide](model_guide.md)** - How to add and integrate new models
* **[Task Guide](task_guide.md)** - Create custom evaluation tasks
* **[Current Tasks](current_tasks.md)** - List of all supported evaluation tasks
* **[Run Examples](run_examples.md)** - Example commands for running evaluations
* **[Caching](caching.md)** - Enable and reload results from the JSONL cache
* **[Version 0.3 Features](lmms-eval-0.3.md)** - Audio evaluation and new features
* **[Throughput Metrics](throughput_metrics.md)** - Understanding performance metrics
* **[MMMU Eval Discrepancy Analysis](mmmu-eval-discrepancy.md)** - Why lmms-eval and VLMEvalKit can report different MMMU scores

## Additional Resources

- [Dataset formatting tools](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/tools) — Scripts for converting datasets into lmms-eval compatible formats.
- [Test suite documentation](../test/README.md) — 292 tests covering every pipeline layer, with prompt stability snapshots for 8 classic benchmarks.
- [GitHub repository](https://github.com/EvolvingLMMs-Lab/lmms-eval) — Source code, issue tracker, and contributing guidelines.
- [Discord community](https://discord.gg/zdkwKUqrPy) — Get help and discuss development.

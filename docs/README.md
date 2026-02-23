# LMMs Eval Documentation

Welcome to the documentation for `lmms-eval` - a unified evaluation framework for Large Multimodal Models!

This framework enables consistent and reproducible evaluation of multimodal models across various tasks and modalities including images, videos, and audio.

## Overview

`lmms-eval` provides:
- Standardized evaluation protocols for multimodal models
- Support for image, video, and audio tasks
- Easy integration of new models and tasks
- Reproducible benchmarking with shareable configurations

Majority of this documentation is adapted from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/)

## Table of Contents

* **[Commands Guide](commands.md)** - Learn about command line flags and options
* **[Quick Start](quickstart.md)** - Evaluate your model in 5 minutes
* **[Model Guide](model_guide.md)** - How to add and integrate new models
* **[Task Guide](task_guide.md)** - Create custom evaluation tasks
* **[Current Tasks](current_tasks.md)** - List of all supported evaluation tasks
* **[Run Examples](run_examples.md)** - Example commands for running evaluations
* **[Caching](caching.md)** - Enable and reload results from the JSONL cache
* **[Version 0.3 Features](lmms-eval-0.3.md)** - Audio evaluation and new features
* **[Version 0.6 Features](lmms-eval-0.6.md)** - Eval-as-a-service, async pipeline, and statistical analysis
* **[Version 0.7 Features](lmms-eval-0.7.md)** - YAML-first runs, reasoning-tag stripping, Lance-backed MINERVA video mode, and skill-based agent workflows
* **[Version 0.8 Features](lmms-eval-0.8.md)** - Unified model evaluation roadmap across understanding and generation for image/video/audio modalities
* **[Throughput Metrics](throughput_metrics.md)** - Understanding performance metrics

## Additional Resources

* For dataset formatting tools, see [lmms-eval tools](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/tools)
* For agent orchestration and reference routing, see [lmms-eval skill guide](../skills/lmms-eval-guide/SKILL.md)
* For non-blocking training integration via HTTP service, see [API server reference](../skills/lmms-eval-guide/references/api-server.md)
* For the latest updates, visit our [GitHub repository](https://github.com/EvolvingLMMs-Lab/lmms-eval)

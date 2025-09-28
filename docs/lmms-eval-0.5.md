# LMMS-Eval v0.5: Update Release

## Introduction

LMMS-Eval v0.5 brings new capabilities for audio evaluation, expanding support for state-of-the-art models and diverse paralinguistic datasets. This release continues to improve multimodal benchmarking, focusing on audio understanding, paralinguistic analysis, and robust evaluation protocols.

## Table of Contents

- [Introduction](#introduction)
- [Major Features](#major-features)
  - [1. New Model: GPT-4o Audio Preview](#1-new-model-gpt-4o-audio-preview)
  - [2. New Datasets](#2-new-datasets)
    - [Step2 Audio Paralinguistic](#step2-audio-paralinguistic)
    - [VoiceBench](#voicebench)
    - [WenetSpeech](#wenetspeech)
  - [3. Enhanced Audio Evaluation Pipeline](#3-enhanced-audio-evaluation-pipeline)
- [Migration Guide](#migration-guide)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Getting Help](#getting-help)

## Major Features

### 1. New Model: GPT-4o Audio Preview

- **GPT-4o Audio Preview** is now integrated as a supported model.
- Enables advanced audio and paralinguistic understanding, leveraging OpenAI's latest multimodal capabilities.
- Unified message interface and judge API compatibility.

### 2. New Datasets

#### Step2 Audio Paralinguistic

- Focuses on paralinguistic features such as emotion, accent, and speaker traits.
- Supports fine-grained evaluation of non-verbal audio cues.

#### VoiceBench

- Benchmark for voice quality, speaker identification, and expressive speech analysis.
- Covers multiple languages and diverse speaker profiles.

#### WenetSpeech

- Large-scale speech recognition dataset.
- Supports ASR evaluation with WER metrics and paralinguistic annotation.

### 3. Enhanced Audio Evaluation Pipeline

- Improved support for loading and processing HuggingFace audio datasets.
- Unified metrics for paralinguistic and ASR tasks (Accuracy, WER, GPT-4 Eval).
- Task grouping and aggregation for multi-subset datasets.
- Compatibility with new model and dataset formats.

## Migration Guide

- Refer to previous migration guides in v0.4 for updating legacy task and model implementations.

## Contributing

- Contributions for new benchmarks, models, and paralinguistic evaluation protocols are welcome.
- See [docs/model_guide.md](docs/model_guide.md) and [docs/task_guide.md](docs/task_guide.md) for implementation details.

## Acknowledgments

- Thanks to all contributors for dataset integration, model support, and pipeline improvements.

## Getting Help

- For issues and support, visit [GitHub Issues](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues).
- Documentation and guides are available in the `docs/` directory.
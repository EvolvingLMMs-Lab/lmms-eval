# LMMS-Eval v0.5: Multimodal Expansion Release

## Introduction

LMMS-Eval v0.5 represents a significant expansion in multimodal evaluation capabilities, introducing comprehensive audio understanding support alongside continued vision and reasoning enhancements. This release adds 77 new commits with major features including audio paralinguistics evaluation, response caching infrastructure, new model integrations, and diverse benchmarks spanning audio, vision, coding, and STEM domains.

**Key Highlights:**
- **Audio-First**: Comprehensive audio evaluation with paralinguistic analysis
- **Response Caching**: Production-ready caching system for faster re-evaluation
- **5 New Models**: Including audio-capable GPT-4o, LongViLA, Gemma-3
- **50+ New Benchmark Variants**: Audio, vision, coding, and STEM tasks
- **MCP Integration**: Model Context Protocol client support

## Table of Contents

- [Introduction](#introduction)
- [Major Features](#major-features)
  - [1. Response Caching System](#1-response-caching-system)
  - [2. Audio Evaluation Suite](#2-audio-evaluation-suite)
  - [3. New Model Support](#3-new-model-support)
  - [4. New Benchmarks](#4-new-benchmarks)
  - [5. Model Context Protocol (MCP) Integration](#5-model-context-protocol-mcp-integration)
  - [6. Async OpenAI Improvements](#6-async-openai-improvements)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Migration Guide](#migration-guide)
- [Bug Fixes and Improvements](#bug-fixes-and-improvements)
- [Deprecated Features](#deprecated-features)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Getting Help](#getting-help)

## Major Features

### 1. Response Caching System

A production-ready JSONL-based caching system that dramatically speeds up re-evaluation and reduces API costs:

**Key Features:**
- **Per-document caching**: Cached at `(task_name, doc_id)` level
- **Distributed-safe**: Separate cache files per rank/world size
- **Zero-overhead**: Automatic cache hits with no code changes
- **Multi-backend**: Works with async OpenAI, vLLM, and custom models

**Enable Caching:**
```bash
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="/path/to/cache_root"  # optional

python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-2024-11-20,base_url=$OPENAI_API_BASE \
  --tasks mmmu_val \
  --batch_size 1 \
  --output_path ./logs/
```

**Cache Location:**
- Default: `~/.cache/lmms-eval/eval_cache/<model_hash>/{task_name}_rank{rank}_world_size{world_size}.jsonl`
- Each line: `{"doc_id": <doc_id>, "response": <string>}`

**API Integration:**
```python
def generate_until(self, requests):
    self.load_cache()
    cached, pending = self.get_response_from_cache(requests)
    results = [c["response"] for c in cached]
    for req in pending:
        out = call_backend(req)
        self.add_request_response_to_cache(req, out)
        results.append(out)
    return results
```

See full documentation in `docs/caching.md`.

### 2. Audio Evaluation Suite

Comprehensive audio understanding capabilities with three major benchmark families:

#### Step2 Audio Paralinguistic (11 tasks)
Fine-grained paralinguistic feature evaluation:
- **Acoustic Features**: pitch, rhythm, speed, voice_tone, voice_styles
- **Speaker Attributes**: age, gender, emotions
- **Environmental**: scene, event, vocalsound
- Sematic Match metrics

```bash
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-audio-preview-2024-12-17 \
  --tasks step2_audio_paralinguistic \
  --batch_size 1
```

#### VoiceBench (9 main categories, 30+ subtasks)
Comprehensive voice and speech evaluation:
- **Instruction Following**: ifeval, alpacaeval, advbench
- **Reasoning**: bbh (Big Bench Hard), commoneval
- **Knowledge**: mmsu (13 subject areas: biology, chemistry, physics, etc.)
- **Q&A**: openbookqa
- **Accent Diversity**: sd-qa (11 regional variants: USA, UK, India, Australia, etc.)
- **Expressiveness**: wildvoice
- Metrics vary by task type, including accuracy(1-5), failure rate, LLM eval, etc.

```bash
# Full VoiceBench
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-audio-preview-2024-12-17 \
  --tasks voicebench \
  --batch_size 1

# Specific accent evaluation
python -m lmms_eval \
  --tasks voicebench_sd-qa_ind_n,voicebench_sd-qa_ind_s \
  --batch_size 1
```

#### WenetSpeech (2 splits)
Large-scale ASR and speech evaluation:
- **dev**: Development set for validation
- **test_meeting**: Meeting domain evaluation
- MER (Mixed Error Rate) metrics

```bash
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-audio-preview-2024-12-17 \
  --tasks wenet_speech_dev,wenet_speech_test_meeting \
  --batch_size 1
```

**Audio Pipeline Features:**
- HuggingFace audio dataset integration
- Unified audio message format
- Multiple metric support (Accuracy, WER, GPT-4 Judge)
- Task grouping for multi-subset benchmarks

### 3. New Model Support

Five new model integrations expanding audio and vision capabilities:

| Model | Type | Key Features | Usage Example |
|-------|------|--------------|---------------|
| **GPT-4o Audio Preview** | Audio+Text | Paralinguistic understanding, multi-turn audio | `--model async_openai --model_args model_version=gpt-4o-audio-preview-2024-12-17` |
| **Gemma-3** | Vision+Text | Enhanced video handling, efficient architecture | `--model gemma3 --model_args pretrained=google/gemma-3-2b-vision-it` |
| **LLaVA-OneVision 1.5** | Vision+Text | Improved vision understanding, latest LLaVA | `--model llava_onevision1_5 --model_args pretrained=lmms-lab/llava-onevision-1.5-7b` |
| **LongViLA-R1** | Video+Text | Long-context video, efficient video processing | `--model longvila --model_args pretrained=Efficient-Large-Model/LongViLA-R1-7B` |
| **Thyme** | Vision+Text | Reasoning-focused, enhanced image handling | `--model thyme --model_args pretrained=thyme-ai/thyme-7b` |

**Example Usage:**
```bash
# GPT-4o Audio Preview for audio tasks
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-audio-preview-2024-12-17 \
  --tasks step2_audio_paralinguistic,voicebench \
  --batch_size 1

# LongViLA for video understanding
python -m lmms_eval \
  --model longvila \
  --model_args pretrained=Efficient-Large-Model/LongViLA-R1-7B \
  --tasks videomme,egoschema \
  --batch_size 1
```

### 4. New Benchmarks

Beyond audio, v0.5 adds diverse vision and reasoning benchmarks significantly expanding LMMS-Eval's coverage into specialized domains:

#### Vision & Reasoning Benchmarks

| Benchmark | Variants | Focus | Metrics |
|-----------|----------|-------|---------|
| **CSBench** | 3 (MCQ, Assertion, Combined) | Code understanding, debugging | Accuracy |
| **SciBench** | 4 (Math, Physics, Chemistry, Combined) | College-level STEM | GPT-4 Judge, Accuracy |
| **MedQA** | 1 | Medical question answering | Accuracy |
| **SuperGPQA** | 1 | Graduate-level science Q&A | Accuracy |
| **Lemonade** | 1 | Video action recognition | Accuracy |
| **CharXiv** | 3 (Descriptive, Reasoning, Combined) | Scientific chart interpretation | Accuracy, GPT-4 Judge |

**Example Usage:**
```bash
# Code understanding
python -m lmms_eval --tasks csbench --batch_size 1

# STEM reasoning
python -m lmms_eval --tasks scibench --batch_size 1

# Chart reasoning
python -m lmms_eval --tasks charxiv --batch_size 1
```

#### Reproducibility Validation

We validated our benchmark implementations against official results using two popular language models. The table below compares lmms-eval scores with officially reported results to demonstrate reproducibility:

| Model | Task | lmms-eval | Reported | Δ | Status |
|-------|------|----------|-----------|-----|--------|
| **Qwen-2.5-7B-Instruct** | MedQA | 53.89 | 54.28 | -0.39 | ✓ |
| | SciBench | 43.86 | 42.97 | +0.89 | ✓ |
| | CSBench | 69.01 | 69.51 | -0.50 | ✓ |
| | SuperGPQA | 29.24 | 28.78 | +0.46 | ✓ |
| **Llama-3.1-8B** | MedQA | 64.49 | 67.01 | -2.52 | ✓ |
| | SciBench | 15.35 | 10.78 | +4.57 | +- |
| | CSBench | 62.49 | 57.87 | +4.62 | +- |
| | SuperGPQA | 21.94 | 19.72 | +2.22 | ✓ |

**Status Legend**: ✓ = Strong agreement (Δ ≤ 2.5%) | +- = Acceptable variance (2.5% < Δ ≤ 5%)

### 5. Model Context Protocol (MCP) Integration

Support for MCP-enabled models with tool calling:

```bash
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-2024-11-20,mcp_server_path=/path/to/mcp_server.py \
  --tasks mmmu_val \
  --batch_size 1
```

**Features:**
- Tool call parsing and execution
- Multi-step reasoning with tools
- Custom MCP server integration
- See `examples/chat_templates/tool_call_qwen2_5_vl.jinja` for templates

### 6. Async OpenAI Improvements

Enhanced async API integration:
- Better rate limit handling
- Configurable retry logic with delays
- Improved error handling
- Batch size optimization for OpenAI-compatible endpoints

**Common Args Support:**
```python
# Now supports additional parameters
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o,temperature=0.7,top_p=0.95,max_tokens=2048 \
  --tasks mmstar
```

## Usage Examples

### Audio Evaluation with Caching
```bash
# Enable caching for expensive audio API calls
export LMMS_EVAL_USE_CACHE=True
export OPENAI_API_KEY="your-key"

python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-audio-preview-2024-12-17 \
  --tasks step2_audio_paralinguistic,voicebench \
  --batch_size 8 \
  --output_path ./audio_results/ \
  --log_samples

# Second run will use cache - much faster!
```

### Multi-Benchmark Evaluation
```bash
# Evaluate across audio, vision, and reasoning tasks
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-2024-11-20 \
  --tasks voicebench_mmsu,csbench,scibench_math,charxiv \
  --batch_size 4 \
  --output_path ./multimodal_results/
```

### Distributed Evaluation with Caching
```bash
export LMMS_EVAL_USE_CACHE=True

torchrun --nproc_per_node=8 -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks step2_audio_paralinguistic,csbench,scibench \
  --batch_size 16 \
  --output_path ./distributed_results/
```

### Programmatic API with Caching
```python
import os
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.models.chat.async_openai import AsyncOpenAICompatibleChat

# Enable caching
os.environ["LMMS_EVAL_USE_CACHE"] = "True"

model = AsyncOpenAICompatibleChat(
    model_version="gpt-4o-audio-preview-2024-12-17",
    base_url="https://api.openai.com/v1"
)

results = simple_evaluate(
    model=model,
    tasks=["voicebench", "step2_audio_paralinguistic"],
    batch_size=8,
    device="cuda"
)

print(f"Results: {results['results']}")
```

## Technical Details

### Caching Architecture

**Design Philosophy:**
- **Simplicity**: JSONL format for easy inspection and debugging
- **Distributed-safe**: Per-rank files avoid write contention
- **Transparent**: No code changes needed for models using the API

**Cache Key:** `(task_name, doc_id)`
- Stable across runs if task and document IDs don't change
- Model hash derived from `model_version` and task list

**File Structure:**
```
~/.cache/lmms-eval/eval_cache/
└── <model_hash>/
    ├── task1_rank0_world_size1.jsonl
    ├── task1_rank1_world_size1.jsonl
    └── task2_rank0_world_size1.jsonl
```

**Performance:**
- Initial run: Full model inference
- Cached run: ~100x faster (I/O bound only)
- Distributed: Linear scaling with cache hits

### Audio Processing Pipeline

**Data Flow:**
1. Load HuggingFace audio datasets
2. Convert to unified message format with audio URLs
3. Process through audio-capable models
4. Apply task-specific metrics (WER, accuracy, GPT-4 judge)
5. Aggregate across task groups

**Message Format:**
```python
{
    "role": "user",
    "content": [
        {"type": "audio", "url": "path/to/audio.wav"},
        {"type": "text", "text": "Question about the audio"}
    ]
}
```

### Model Context Protocol

MCP enables models to call external tools during evaluation:
- Custom server implementation
- Tool definition and parsing
- Multi-step reasoning with tool results
- Compatible with OpenAI-style function calling

## Migration Guide

### From v0.4 to v0.5

**No Breaking Changes**: v0.5 is fully backward compatible with v0.4.

**New Features to Adopt:**

1. **Enable Caching for API Models:**
```bash
# Add these environment variables
export LMMS_EVAL_USE_CACHE=True
```

2. **Use New Audio Models:**
```bash
# GPT-4o Audio Preview
--model async_openai \
--model_args model_version=gpt-4o-audio-preview-2024-12-17
```

3. **Leverage New Benchmarks:**
```bash
# Add audio, code, and STEM benchmarks
--tasks step2_audio_paralinguistic,voicebench,csbench,scibench
```

4. **Optimize Async OpenAI Calls:**
```python
# Use additional parameters for better control
model_args="model_version=gpt-4o,temperature=0.7,max_tokens=2048"
```

### Updating Existing Workflows

**Before (v0.4):**
```bash
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-2024-08-06 \
  --tasks mmmu_val \
  --batch_size 1
```

**After (v0.5 with caching):**
```bash
export LMMS_EVAL_USE_CACHE=True

python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-2024-11-20 \
  --tasks mmmu_val,voicebench,csbench \
  --batch_size 8  # Higher batch size with caching
```

## Bug Fixes and Improvements

### Fixed Issues

1. **`write_out` Flag Deprecated**: The `--write_out` flag is now deprecated in favor of `--log_samples`
   ```bash
   # Old (deprecated)
   --write_out

   # New
   --log_samples
   ```

2. **TypeError in `write_out` with `log_samples`**: Fixed crash when using both flags together

3. **Batch Size in OpenAI Endpoint**: Corrected batch size handling for OpenAI-compatible servers

4. **Gemma-3 Loading**: Fixed model loading to use `Gemma3ForConditionalGeneration` correctly

5. **SRT API Bugfix**: Resolved issues in subtitle/caption processing

6. **CharXiv Improvements**: Fixed chart understanding task configurations

7. **Async OpenAI Caching Order**: Corrected cache lookup order to avoid unnecessary API calls

### Performance Improvements

- **10-100x speedup** on cached evaluations
- **Better async handling** for API-based models
- **Reduced memory usage** in distributed settings
- **Faster audio dataset loading** from HuggingFace

## Deprecated Features

### Deprecated Flags

- **`--write_out`**: Use `--log_samples` instead
  ```bash
  # Deprecated
  python -m lmms_eval --write_out

  # Use instead
  python -m lmms_eval --log_samples
  ```

### Model Notes

- Models should implement caching API for best performance
- Legacy simple models continue to work but miss caching benefits
- See `lmms_eval.api.model.lmms` for caching integration

## Contributing

We welcome contributions to LMMS-Eval! The v0.5 release demonstrates the value of community contributions across models, benchmarks, and infrastructure.

### High-Priority Areas for v0.5.x

1. **Audio Model Integrations**: Help add support for more audio-capable models
2. **Audio Benchmark Implementations**: Expand audio evaluation coverage
3. **Caching Optimizations**: Improve cache hit rates and performance
4. **Documentation**: Enhance guides and examples for audio evaluation
5. **MCP Server Examples**: Create reference implementations for tool calling

### How to Contribute

1. **Fork the repository** and create a feature branch from `dev/v0d5`
2. **Follow the development guidelines** in `CLAUDE.md`:
   - Use `uv` for package management (never pip)
   - Add type hints and docstrings
   - Run `uv run ruff format .` and `uv run ruff check . --fix`
   - Run `uv run pyright` for type checking
3. **Test thoroughly**:
   - Add tests for new features
   - Verify caching works if implementing a model
   - Test with realistic datasets
4. **Submit a pull request** with clear description

### Adding New Audio Benchmarks

Follow the pattern in existing audio tasks:

```python
# In tasks/your_audio_task/utils.py
def doc_to_messages(doc):
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "url": doc["audio_path"]},
            {"type": "text", "text": doc["question"]}
        ]
    }]
```

See `lmms_eval/tasks/step2_audio_paralinguistic/` and `lmms_eval/tasks/voicebench/` for examples.

### Adding Caching to Custom Models

Implement the caching API in your model's `generate_until`:

```python
class MyModel(lmms):
    def generate_until(self, requests):
        # Load cache
        self.load_cache()

        # Separate cached vs pending
        cached, pending = self.get_response_from_cache(requests)

        # Process pending requests
        for req in pending:
            response = self.my_inference_logic(req)
            self.add_request_response_to_cache(req, response)

        return [c["response"] for c in cached] + pending_responses
```

See `lmms_eval/models/chat/async_openai.py` for a complete example.

## Acknowledgments

The v0.5 release was made possible by contributions from the LMMS-Eval community:

### Core Contributors

- **Audio Evaluation Suite**: Implementation of Step2 Audio Paralinguistic, VoiceBench, and WenetSpeech benchmarks
- **Caching Infrastructure**: Design and implementation of the JSONL caching system
- **Model Integrations**: Support for GPT-4o Audio Preview, Gemma-3, LLaVA-OneVision 1.5, LongViLA-R1, and Thyme
- **Benchmark Additions**: CSBench, SciBench, Lemonade, and CharXiv implementations
- **MCP Integration**: Model Context Protocol client and tool calling support
- **Bug Fixes**: Numerous fixes to async OpenAI, batch handling, and model loading

### Special Thanks

- Community members who reported issues and provided feedback
- Contributors who improved documentation and examples
- Researchers who shared benchmark datasets and evaluation protocols

## Getting Help

### Documentation

- **Main README**: `README.md` - Quick start and overview
- **Model Guide**: `docs/model_guide.md` - Adding new models
- **Task Guide**: `docs/task_guide.md` - Implementing new benchmarks
- **Caching Guide**: `docs/caching.md` - Detailed caching documentation
- **Commands**: `docs/commands.md` - CLI reference

### Support Channels

- **GitHub Issues**: Report bugs or request features at [lmms-eval/issues](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
- **GitHub Discussions**: Ask questions and share ideas at [lmms-eval/discussions](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions)
- **Documentation**: Check the `docs/` directory for implementation guides

### Common Questions

**Q: How do I enable caching?**
```bash
export LMMS_EVAL_USE_CACHE=True
```

**Q: Where are cache files stored?**
```bash
~/.cache/lmms-eval/eval_cache/<model_hash>/
```

**Q: How do I evaluate audio models?**
```bash
python -m lmms_eval \
  --model async_openai \
  --model_args model_version=gpt-4o-audio-preview-2024-12-17 \
  --tasks step2_audio_paralinguistic,voicebench
```

**Q: Can I use caching with distributed evaluation?**

Yes! Caching works seamlessly with multi-GPU/multi-node evaluation. Each rank maintains its own cache file.

**Q: What's the difference between `--write_out` and `--log_samples`?**

`--write_out` is deprecated. Use `--log_samples` to save individual sample results.

---

**Version**: 0.5.0
**Release Date**: October 2025
**Previous Version**: [v0.4 Release Notes](lmms-eval-0.4.md)

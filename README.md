<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Probing Intelligence in the Real World

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> We are building the unified evaluation toolkit for frontier models and probing the abilities in real world, shape what we build next.

<details>
<summary>🌐 Available in 17 languages</summary>

[简体中文](docs/i18n/README_zh-CN.md) | [繁體中文](docs/i18n/README_zh-TW.md) | [日本語](docs/i18n/README_ja.md) | [한국어](docs/i18n/README_ko.md) | [Español](docs/i18n/README_es.md) | [Français](docs/i18n/README_fr.md) | [Deutsch](docs/i18n/README_de.md) | [Português](docs/i18n/README_pt-BR.md) | [Русский](docs/i18n/README_ru.md) | [Italiano](docs/i18n/README_it.md) | [Nederlands](docs/i18n/README_nl.md) | [Polski](docs/i18n/README_pl.md) | [Türkçe](docs/i18n/README_tr.md) | [العربية](docs/i18n/README_ar.md) | [हिन्दी](docs/i18n/README_hi.md) | [Tiếng Việt](docs/i18n/README_vi.md) | [Indonesia](docs/i18n/README_id.md)

</details>

📚 [Documentation](docs/README.md) | 📖 [100+ Tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [30+ Models](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | ⚡ [Quickstart](docs/quickstart.md)

🏠 [Homepage](https://www.lmms-lab.com/) | 💬 [Discord](https://discord.gg/8xTM6jWnXa) | 🤝 [Contributing](CONTRIBUTING.md)

---

## Why `lmms-eval`?

We are on a journey toward Artificial General Intelligence, similar to the excitement of the moon landing era. This effort is driven by large language models and multimodal models that can understand and act across many kinds of tasks. To measure progress, we rely on benchmarks. But the ecosystem is fragmented. Datasets are scattered across different platforms, each with its own format, scripts, and post processing logic. Two teams evaluating the same model on the same benchmark often report different numbers. The difference often comes from evaluation details.

Even with a unified pipeline, deeper problems remain. Existing open source evaluation toolkits are packaged as libraries, not as services that can be used in different scenarios (async eval call during training, or a job submission after training in standalone cluster). And scores are usually reported as a single accuracy number, representing only the mean from a statistical perspective. It is hard to tell whether a small gain is a true improvement without estimating variance and considering the number of samples in each benchmark. There are many more to be fixed and explored. [lmms-eval v0.6](docs/lmms-eval-0.6.md) is a small step towards better eval toolkit.

We are keeping the evolving pace to build better evals, to explore evaluation's role on the path to frontier models. We believe [better evals lead to better models](https://arxiv.org/pdf/2211.09110): good evaluation maps the border of model capabilities and shapes what we build next. That is what it means to probe intelligence in the real world.

## What's New

**February 2026 (v0.6)** - Our previous versions were too slow, the architecture wasn't clean, and the results lacked statistical insight. v0.6 is a re-engineered release that addresses all three: evaluation runs as a standalone service (decoupled from training, serving queue-based eval requests), statistically grounded results that capture real model improvements rather than a single accuracy score (confidence intervals, clustered standard errors, paired comparison with t-test), and optimizations to max out your model runtime's capacity (~7.5x over previous versions). 50+ new tasks and 10+ new models. [Release notes](docs/lmms-eval-0.6.md) | [Changelog](CHANGELOG.md).

**October 2025 (v0.5)** - Audio had been a gap. Models could hear, but we had no consistent way to test them. This release added comprehensive audio evaluation, response caching for efficiency, and 50+ benchmark variants spanning audio, vision, and reasoning. [Release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).

<details>
<summary>Below is a chronological list of recent tasks, models, and features added by our amazing contributors. </summary>

- [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
- [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
- [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
- [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
- [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
- [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)!
- [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

</details>

## Quickstart

Install and run your first evaluation in under 5 minutes:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && uv pip install -e ".[all]"

# Run a quick evaluation (Qwen2.5-VL on MME, 8 samples)
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

If it prints metrics, your environment is ready. For the full guide, see [`docs/quickstart.md`](docs/quickstart.md).

## Installation

### Using `uv` (Recommended for consistent environments)

We use `uv` for package management to ensure all developers use exactly the same package versions. First, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For development with consistent environment:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Recommend
uv pip install -e ".[all]"
# If you want to use uv sync
# uv sync  # This creates/updates your environment from uv.lock
```

To run commands:
```bash
uv run python -m lmms_eval --help  # Run any command with uv run
```

To add new dependencies:
```bash
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

For direct usage from Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# You might need to add and include your own task yaml if using this installation
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproduction of LLaVA-1.5's paper results</summary>

You can check the [torch environment info](miscs/repr_torch_envs.txt) and [results check](miscs/llava_result_check.md) to **reproduce LLaVA-1.5's paper results**. We found torch/cuda versions difference would cause small variations in the results.

</details>

If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version`


<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

Our Development will be continuing on the main branch, and we encourage you to give us feedback on what features are desired and how to improve the library further, or ask questions, either in issues or PRs on GitHub.

## Usage Examples

> More examples can be found in [examples/models](examples/models)

### Evaluation with vLLM

**Qwen2.5-VL:**
```bash
bash examples/models/vllm_qwen2vl.sh
```

**Qwen3-VL:**
```bash
bash examples/models/vllm_qwen3vl.sh
```

**Qwen3.5:**
```bash
bash examples/models/vllm_qwen35.sh
```

### Evaluation with SGLang

```bash
bash examples/models/sglang.sh
```

**Qwen3.5:**
```bash
bash examples/models/sglang_qwen35.sh
```

### Evaluation of OpenAI-Compatible Model

```bash
bash examples/models/openai_compatible.sh
```

### Evaluation of Qwen2.5-VL

```bash
bash examples/models/qwen25vl.sh
```

### Evaluation of Qwen3-VL

```bash
bash examples/models/qwen3vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

**Environmental Variables**

Before running experiments and evaluations, we recommend you to export following environment variables to your environment. Some are necessary for certain tasks to run.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Environment Issues**

Sometimes you might encounter some common issues for example error related to httpx or protobuf. To solve these issues, you can first try

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

## Custom Model Integration

`lmms-eval` supports two types of models: **Chat (recommended)** and **Simple (legacy)**.

### Chat Models (Recommended) 🌟

- Location: `lmms_eval/models/chat/`
- Use: `doc_to_messages` function from task
- Input: Structured `ChatMessages` with roles (`user`, `system`, `assistant`) and content types (`text`, `image`, `video`, `audio`)
- Supports: Interleaved multimodal content
- Uses: Model's `apply_chat_template()` method
- Reference: `lmms_eval/models/chat/qwen2_5_vl.py` or `lmms_eval/models/chat/qwen3_vl.py`

**Example input format:**
```python
[
    {"role": "user", "content": [
        {"type": "image", "url": <image>},
        {"type": "text", "text": "What's in this image?"}
    ]}
]
```

### Simple Models (Legacy)

- Location: `lmms_eval/models/simple/`
- Use: `doc_to_visual` + `doc_to_text` functions from task
- Input: Plain text with `<image>` placeholders + separate visual list
- Supports: Limited (mainly images)
- Manual processing: No chat template support
- Reference: `lmms_eval/models/simple/instructblip.py`

**Example input format:**
```python
# Separate visual and text
doc_to_visual -> [PIL.Image]
doc_to_text -> "What's in this image?"
```

### Key Differences

| Aspect | Chat Models | Simple Models |
|--------|-------------|---------------|
| File location | `models/chat/` | `models/simple/` |
| Input method | `doc_to_messages` | `doc_to_visual` + `doc_to_text` |
| Message format | Structured (roles + content types) | Plain text with placeholders |
| Interleaved support | ✅ Yes | ❌ Limited |
| Chat template | ✅ Built-in | ❌ Manual/None |
| Recommendation | **Use this** | Legacy only |

### Why Use Chat Models?

- ✅ Built-in chat template support
- ✅ Interleaved multimodal content
- ✅ Structured message protocol
- ✅ Better video/audio support
- ✅ Consistent with modern LLM APIs

### Chat Model Implementation Example

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.protocol import ChatMessages

@register_model("my_chat_model")
class MyChatModel(lmms):
    is_simple = False  # Use chat interface

    def generate_until(self, requests):
        for request in requests:
            # 5 elements for chat models
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            # Get structured messages
            raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
            messages = ChatMessages(messages=raw_messages)

            # Extract media and apply chat template
            images, videos, audios = messages.extract_media()
            hf_messages = messages.to_hf_messages()
            text = self.processor.apply_chat_template(hf_messages)

            # Generate...
```

For more details, see the [Model Guide](docs/model_guide.md).

## Custom Dataset Integration

### Task Configuration with `doc_to_messages`

Implement `doc_to_messages` to transform dataset documents into structured chat messages:

```python
def my_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    # Extract visuals and text from doc
    visuals = my_doc_to_visual(doc)
    text = my_doc_to_text(doc, lmms_eval_specific_kwargs)

    # Build structured messages
    messages = [{"role": "user", "content": []}]

    # Add visuals first
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})

    # Add text
    messages[0]["content"].append({"type": "text", "text": text})

    return messages
```

### YAML Configuration

```yaml
task: "my_benchmark"
dataset_path: "my-org/my-dataset"
test_split: test
output_type: generate_until

# For chat models (recommended)
doc_to_messages: !function utils.my_doc_to_messages

# OR legacy approach:
doc_to_visual: !function utils.my_doc_to_visual
doc_to_text: !function utils.my_doc_to_text

process_results: !function utils.my_process_results
metric_list:
  - metric: acc
```

### Key Features

#### `doc_to_messages`

- Transforms dataset document into structured chat messages
- Returns: List of message dicts with `role` and `content`
- Content supports: `text`, `image`, `video`, `audio` types
- Protocol: Defined in `lmms_eval/protocol.py` (`ChatMessages` class)
- Auto-fallback: If not provided, uses `doc_to_visual` + `doc_to_text`


For more details, see the [Task Guide](docs/task_guide.md).

## Web UI

LMMS-Eval includes an optional Web UI for interactive evaluation configuration.

### Requirements

- Node.js 18+ (for building the frontend, auto-built on first run)

### Usage

```bash
# Start the Web UI (opens browser automatically)
uv run lmms-eval-ui

# Custom port
LMMS_SERVER_PORT=3000 uv run lmms-eval-ui
```

The web UI provides:
- Model selection from all available models
- Task selection with search/filter
- Real-time command preview
- Live evaluation output streaming
- Start/Stop evaluation controls

For more details, see [Web UI README](lmms_eval/tui/README.md).

## HTTP Evaluation Server

LMMS-Eval includes a production-ready HTTP server for remote evaluation workflows.

### Why Use Eval Server?

- **Decoupled evaluation**: Run evaluations on dedicated GPU nodes while training continues
- **Async workflow**: Submit jobs without blocking training loops
- **Queue management**: Sequential job processing with automatic resource management
- **Remote access**: Evaluate models from any machine

### Start Server

```python
from lmms_eval.entrypoints import ServerArgs, launch_server

# Configure server
args = ServerArgs(
    host="0.0.0.0",
    port=8000,
    max_completed_jobs=200,
    temp_dir_prefix="lmms_eval_"
)

# Launch server
launch_server(args)
```

Server runs at `http://host:port` with auto-generated API docs at `/docs`

### Client Usage

**Sync Client:**
```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")

# Submit evaluation (non-blocking)
job = client.evaluate(
    model="qwen2_5_vl",
    tasks=["mmmu_val", "mme"],
    model_args={"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"},
    num_fewshot=0,
    batch_size=1,
    device="cuda:0",
)

# Continue training...
# Later, retrieve results
result = client.wait_for_job(job["job_id"])
print(result["result"])
```

**Async Client:**
```python
from lmms_eval.entrypoints import AsyncEvalClient

async with AsyncEvalClient("http://eval-server:8000") as client:
    job = await client.evaluate(
        model="qwen3_vl",
        tasks=["mmmu_val"],
        model_args={"pretrained": "Qwen/Qwen3-VL-4B-Instruct"},
    )
    result = await client.wait_for_job(job["job_id"])
```

### Server API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/evaluate` | POST | Submit evaluation job |
| `/jobs/{job_id}` | GET | Get job status and results |
| `/queue` | GET | View queue status |
| `/tasks` | GET | List available tasks |
| `/models` | GET | List available models |
| `/jobs/{job_id}` | DELETE | Cancel queued job |
| `/merge` | POST | Merge FSDP2 sharded checkpoints |

### Example Workflow

```python
# Training loop pseudocode
for epoch in range(num_epochs):
    train_one_epoch()

    # After every N epochs, evaluate checkpoint
    if epoch % 5 == 0:
        checkpoint_path = f"checkpoints/epoch_{epoch}"

        # Submit async evaluation (non-blocking)
        eval_job = client.evaluate(
            model="vllm",
            model_args={"model": checkpoint_path},
            tasks=["mmmu_val", "mathvista"],
        )

        # Training continues immediately
        print(f"Evaluation job submitted: {eval_job['job_id']}")

# After training completes, retrieve all results
results = []
for job_id in eval_jobs:
    result = client.wait_for_job(job_id)
    results.append(result)
```

### Security Note

⚠️ **This server is intended for trusted environments only**. Do NOT expose to untrusted networks without additional security layers (authentication, rate limiting, network isolation).

For more details, see the [v0.6 release notes](docs/lmms-eval-0.6.md).

## Frequently Asked Questions

<details>
<summary><strong>What models does lmms-eval support?</strong></summary>

We support 30+ model families out of the box, including Qwen2.5-VL, Qwen3-VL, LLaVA-OneVision, InternVL-2, VILA, and more. Any OpenAI-compatible API endpoint is also supported. See the full list in [`lmms_eval/models/`](lmms_eval/models/).

Qwen3.5 is supported through existing runtime backends (`--model vllm` and `--model sglang`) by setting `model=Qwen/Qwen3.5-397B-A17B` in `--model_args`.

The Qwen3.5 example scripts align with official runtime references (for example, `max_model_len/context_length=262144` and `reasoning_parser=qwen3`).

If a new model family is already fully supported by vLLM or SGLang at runtime, we generally only need documentation and examples instead of adding a dedicated model wrapper.

</details>

<details>
<summary><strong>What benchmarks and tasks are available?</strong></summary>

Over 100 evaluation tasks across image, video, and audio modalities, including MMMU, MME, MMBench, MathVista, VideoMME, EgoSchema, and many more. Check [`docs/current_tasks.md`](docs/current_tasks.md) for the full list.

</details>

<details>
<summary><strong>How do I add my own benchmark?</strong></summary>

Create a YAML config under `lmms_eval/tasks/` with dataset path, splits, and a `doc_to_messages` function. See [`docs/task_guide.md`](docs/task_guide.md) for a step-by-step guide.

</details>

<details>
<summary><strong>Can I evaluate a model behind an API (e.g., GPT-4o, Claude)?</strong></summary>

Yes. Use `--model openai` with `--model_args model=gpt-4o` and set `OPENAI_API_KEY`. Any OpenAI-compatible endpoint works, including local vLLM/SGLang servers.

</details>

<details>
<summary><strong>How do I run evaluations on multiple GPUs?</strong></summary>

Use `accelerate launch` or pass `--device cuda` with tensor parallelism via vLLM/SGLang backends. See [`docs/commands.md`](docs/commands.md) for multi-GPU flags.

</details>

<details>
<summary><strong>How do I cite lmms-eval?</strong></summary>

Use the BibTeX entries below, or click the "Cite this repository" button in the GitHub sidebar (powered by our [`CITATION.cff`](CITATION.cff)).

</details>

## Acknowledgement

lmms_eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend you to read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant information.

---

Below are the changes we made to the original API:
- Build context now only pass in idx and process image and doc during the model responding phase. This is due to the fact that dataset now contains lots of images and we can't store them in the doc like the original lm-eval-harness otherwise the cpu memory would explode.
- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF language models as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Therefore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.

---

## Citations

```bibtex
@misc{zhang2024lmmsevalrealitycheckevaluation,
      title={LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models},
      author={Kaichen Zhang and Bo Li and Peiyuan Zhang and Fanyi Pu and Joshua Adrian Cahyono and Kairui Hu and Shuai Liu and Yuanhan Zhang and Jingkang Yang and Chunyuan Li and Ziwei Liu},
      year={2024},
      eprint={2407.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12772},
}

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}
```

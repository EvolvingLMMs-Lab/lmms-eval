# LMMs-Eval v0.6

## Overview

After developing LMMs-Eval for over a year, we've integrated 100+ tasks and 30+ models across images, videos, and audio. Throughout this journey, we've grown increasingly aware that **evaluation itself deserves the same rigor we demand from the models we evaluate**. A benchmark that cannot reliably indicate a model's true capabilities is not just unhelpful, it actively misleads research directions.

This realization drives v0.6: a re-architecture designed to make evaluation **fast enough to iterate**, **rigorous enough to trust**, and **challenging enough to matter**.

Building on lmms-eval's existing framework, v0.6 transforms evaluation from a one-off script into a **production-ready evaluation system**. This enables two critical workflows:

- **During training**: Evaluation runs as a standalone service, decoupled from the training loop. Submit checkpoints for async evaluation without blocking GPU training.
- **Post-training**: Rapid, comprehensive evaluation across all modalities with statistical guarantees on the results.

| Area | Key Features |
|------|--------------|
| **Performance** | Fully async and decoupled inference; data layer optimization for high-throughput multimodal access |
| **Evaluation as a Service** | Async job submission without blocking GPU training; separately hosted eval service on dedicated GPUs |
| **Statistical Rigor** | Confidence intervals, clustered standard errors, baseline-anchored paired comparison |
| **Frontier Evaluation** | Long video, spatial intelligence, and agentic scenarios |

---

## 1. Architecture

### 1.1 Inference Backend

#### Decoupled Design

v0.6 separates evaluation logic from model inference. Models implement a standardized `LMM` abstract base class:

```python
class LMM(ABC):
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Batched text generation."""
        pass

    def loglikelihood(self, requests: list[Instance]) -> list[float]:
        """Token-level log probabilities for multiple-choice ranking."""
        pass
```

The `Instance` object carries:
- `media_list`: Raw images, video paths, or audio buffers
- `prompt`: Text prompt
- `gen_kwargs`: Generation parameters

#### Backend Integration

Supported backends:
- vLLM
- SGLang
- API Models (OpenAI, Anthropic, Groq, etc.)

```bash
python -m lmms_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-VL-7B
python -m lmms_eval --model sglang --model_args pretrained=Qwen/Qwen2.5-VL-7B
```

### 1.2 Data Layer

#### Storage Format Migration

| Format | Random Access | Media Handling | Use in v0.6 |
|--------|---------------|----------------|-------------|
| JSON/Files | O(N) scan | External files | Legacy |
| Parquet | Row-group decompress | Binary blobs | Metadata |
| Lance | O(1) lookup | Native blob support | Multimodal data |

**Parquet**: Task metadata (questions, answers, splits). Supports projection pushdown, read only required columns. Underlying format for HF Datasets (disk).

**Lance**: Images and video data. Zero-copy memory mapping via Apache Arrow layout.

#### Why Lance for Multimodal Data?

| Approach | Problem |
|----------|---------|
| Separate image files | Millions of small file I/O, slow metadata lookups |
| Parquet with binary columns | No random access within row groups, full decompression required |

Lance addresses these with a columnar format designed for ML workloads:

| Feature | Mechanism | Benefit |
|---------|-----------|---------|
| O(1) random access | Global offsets & structural encoding | Instant lookup by index |
| Native blob support | Variable-length binary columns | Images/videos inline, no external files |
| Zero-copy reads | Memory-mapped Arrow buffers | No serialization overhead |
| Append-only versioning | Immutable fragments with manifest | Safe concurrent writes |

**Usage Pattern**

```python
import lancedb

db = lancedb.connect("./benchmark_data")
table = db.open_table("videomme")

# Filtered query - only reads necessary columns/rows
samples = (
    table.search()
    .where("split = 'test' AND duration_sec > 60")
    .select(["video_id", "question", "video_bytes"])
    .limit(1000)
    .to_arrow()  # Returns pyarrow.Table (memory-mapped)
)

for batch in samples.to_batches():
    # PyArrow Array backed by memory-mapped buffer
    video_col = batch["video_bytes"]
    # Use video_col[i].as_buffer() to avoid Python bytes copy
```

### 1.3 Evaluation Pipeline

#### Three Components

v0.6 defines evaluation as three decoupled components:

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│  Input  │ ───▶ │  Model  │ ───▶ │  Judge  │
└─────────┘      └─────────┘      └─────────┘
```

| Component | Contents | Output |
|-----------|----------|--------|
| **Input** | Multimodal data + question + ground truth | `Instance` objects |
| **Model** | LMM inference (local or API) | Generated responses |
| **Judge** | Metric computation (exact match, LLM judge, etc.) | Scores |

#### Async Pipeline with Cache

Both stages run asynchronously with intelligent caching:

```
Input ──async──▶ Model ──cache──▶ Storage ──async──▶ Judge
                          │
                          └── Cache key: hash(input + config + git_commit)
```

**Cache System**

Avoid redundant inference when the same evaluation has been run before:

| Cache Key Component | Purpose |
|---------------------|---------|
| Input hash | Same dataset + question |
| Config hash | Same generation parameters (temp, max_tokens, etc.) |
| Git commit | Same model code version |

Cache hit -> skip inference, reuse stored outputs. Cache miss -> run inference, store results.

**Benefits**:
- No redundant computation: identical runs return cached results instantly
- Crash recovery: resume from cached outputs without re-inference
- Resource separation: Model (GPU) and Judge (API) can run on different machines
- Reproducibility: cache key ensures exact same conditions

#### Stage 1: Input -> Model

**Prefix Cache Optimization**

vLLM/SGLang reuse KV cache for shared prefixes. We cluster inputs by media to maximize hits:

- Prefix clustering: Group questions by shared image/video
- Length sorting: Similar lengths in same batch
- Chunked prefill: Split long prompts into chunks to reduce TTFT and avoid OOM

#### Stage 2: Model -> Judge

**Persist-First Strategy**: Save model outputs to disk immediately, then score asynchronously.

### 1.4 Evaluation as a Service

To integrate evaluation into training workflows, v0.6 provides a disaggregated HTTP service architecture.

```
┌─────────────────┐          ┌─────────────────┐           ┌─────────────────┐
│  Training Loop  │ ──POST──▶│   Eval Server   │ ──queue──▶│   Job Worker    │
│   (any host)    │◀──poll── │   (FastAPI)     │◀──result──│   (GPU node)    │
└─────────────────┘          └─────────────────┘           └─────────────────┘
```

**Key benefit**: Training continues while evaluation runs asynchronously on separate resources.

#### Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluate` | POST | Submit evaluation job (model, tasks, config) |
| `/jobs/{job_id}` | GET | Query job status and results |
| `/queue` | GET | View pending/running/completed jobs |
| `/tasks` | GET | List available evaluation tasks |
| `/models` | GET | List supported model backends |

#### Client Usage

```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")

# Submit evaluation (non-blocking)
job = client.evaluate(
    model="qwen2_5_vl",
    tasks=["mmmu_val", "mme"],
    model_args={"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"},
)

# Continue training...
# Later, retrieve results
result = client.wait_for_job(job["job_id"])
```

The server uses a `JobScheduler` that queues requests and processes them sequentially, ensuring proper GPU resource management without conflicts.

---

## 2. Statistical Analysis

### 2.1 Why Statistical Analysis?

Current leaderboards rank models by mean accuracy without uncertainty quantification. As [Anthropic's research](https://www.anthropic.com/research/statistical-approach-to-model-evals) demonstrates, this is fundamentally flawed:

| Problem | Example | Consequence |
|---------|---------|-------------|
| Scores are estimates, not truth | 85% on 1000 questions ≠ 85% true capability | False confidence in rankings |
| Small differences are noise | 85.2% vs 85.5% is statistically insignificant | Wasted effort chasing noise |
| Correlated questions inflate precision | 10 questions per video ≠ 10 independent samples | Underestimated uncertainty |

**The fix**: Treat evaluation as a sampling experiment. Report confidence intervals. Use clustered standard errors. Compare models with paired tests.

**What v0.6 Adds**

| Feature | Purpose |
|---------|---------|
| Standard Error (SE) | Quantify uncertainty of single model score |
| Confidence Intervals | Report score ± margin, not point estimate |
| Clustered SE | Correct for correlated questions (same video/image) |
| Paired Comparison | Detect small differences by removing question difficulty variance |
| Model Stability | Measure inherent variance under standard settings |

### 2.2 Standard Error Estimation

#### Independent Samples

For binary metrics (pass/fail), SE simplifies to:

$$SE = \sqrt{\frac{p(1-p)}{n}}$$

where $p$ = accuracy, $n$ = number of questions.

**Key insight**: SE ∝ $1/\sqrt{n}$. To halve uncertainty -> 4× more questions.

Output format: `score ± 1.96 × SE` (95% CI)

#### Clustered Samples

**Problem**: Multiple questions per video/image are correlated, not independent.

**Solution**: Specify `cluster_key` -> system applies cluster-robust SE correction.

```yaml
task: videomme
cluster_key: video_id
```

Clustered SE can be 3× larger than naive estimates.

### 2.3 Model Comparison

**Problem**: Checking if confidence intervals overlap is low-power.

**Solution**: Paired test — compute per-question difference $d_i = score_A - score_B$, then test if mean($d$) ≠ 0.

**Why**: Removes question difficulty variance (dominant noise), isolates model difference signal.

#### Baseline-Anchored Evaluation

A practical application of paired comparison: **anchor evaluations to a standard baseline model** (e.g., Gemini 3.0 Pro).

| Approach | Report | Limitation |
|----------|--------|------------|
| Absolute score | "Our model: 78.3%" | Meaningless without context |
| Leaderboard rank | "#3 on MMMU" | Rank doesn't quantify gap |
| **Paired difference** | "+2.1% vs Gemini 3.0 Pro (p<0.01)" | Statistically grounded claim |

**Benefits**:
- **Reproducible claims**: "We beat baseline X by Y%" is verifiable
- **Training signal**: Track improvement over baseline across checkpoints
- **Publication-ready**: Statistical significance replaces hand-waving

```python
# Example: Compare your model against Gemini 3.0 Pro baseline
results = paired_comparison(
    model_a="your_model",
    model_b="gemini-3.0-pro",  # Standard baseline
    tasks=["mmmu_val", "mathvista"],
)
# Output: mean_diff=+2.1%, CI=[+0.8%, +3.4%], p=0.003
```

### 2.4 Power Analysis

**Purpose**: Minimum sample size to detect a given effect (e.g., 2% improvement).

**Rule of thumb**: Reliable benchmarks need $n > 1000$ questions.

### 2.5 Model Stability Measurement

#### Why Measure Variance?

Two models with 80% accuracy can behave differently:
- **Model A**: Answers consistently (same questions right/wrong each run)
- **Model B**: Answers randomly (different results each run)

**Model A is more reliable.** Model B's accuracy is "luck."

**Goal**: Measure model's **inherent stability** under standard settings (temp=0.7).

#### Law of Total Variance

$$Var(Score) = \underbrace{Var_{within}}_{\text{Model instability}} + \underbrace{Var_{between}}_{\text{Question difficulty}}$$

The first term measures **model stability** — lower is better.

#### Protocol

Run N samples per question (temp=0.7), report:

| Metric | Meaning |
|--------|---------|
| **Expected Accuracy (EA)** | Mean accuracy across all N samples |
| **Consensus Accuracy (CA)** | Accuracy after majority vote |
| **Internal Variance (IV)** | Model instability — **lower is better** |
| **Consistency Rate (CR)** | % questions with same answer across N runs |

#### Example Output

```
┌─────────────┬─────┬─────┬───────┐
│ Model       │ EA  │ CA  │  IV   │
├─────────────┼─────┼─────┼───────┤
│ Model A     │ 80% │ 82% │ 0.05  │  ← Stable
│ Model B     │ 80% │ 81% │ 0.15  │  ← Unstable
└─────────────┴─────┴─────┴───────┘
```

Same accuracy, but Model A is 3× more stable.

#### Question-Level Diagnostics

| Pattern | Possible Cause |
|---------|----------------|
| High IV across all models | Ambiguous question |
| High IV for one model | Model-specific weakness |
| Zero IV, always wrong | Confidently wrong knowledge |

---

## 3. Frontier Multimodal Evaluation

### 3.1 Why Frontier Scenarios Matter

Static image QA benchmarks are saturating. Building frontier multimodal systems requires setting more challenging tasks and evaluating them in more realistic scenarios:

| Capability | Challenge | Current Gap |
|------------|-----------|-------------|
| Long video understanding | 10min+ videos, 1000+ frames | Most benchmarks use <128 frames |
| High temporal resolution | Event detection at 30fps | Sparse sampling loses fine-grained actions |
| Spatial reasoning | 3D world understanding | 2D perception ≠ physical grounding |
| Agentic interaction (streaming input) | Multi-step task execution | Static QA can't measure planning/tool use |

**Key insight**: These capabilities require **in-environment evaluation**, the model must interact with simulators, receive feedback, and adapt. Static input-output pairs cannot capture this.

### 3.2 Long Video & High Frame Rate

#### The Scale Problem

| Scenario | Frames | Tokens (est.) | Challenge |
|----------|--------|---------------|-----------|
| 1min video @ 1fps | 60 | ~60K | Fits context |
| 10min video @ 1fps | 600 | ~600K | Exceeds most context windows |
| 1min video @ 30fps | 1800 | ~1.8M | Memory explosion |

Streaming metrics:
- **Event detection latency**: Time from event occurrence to model detection
- **Memory efficiency**: Performance vs. KV cache size
- **Graceful degradation**: Accuracy when forced to evict old context

### 3.3 Spatial Intelligence

Spatial intelligence benchmarks are needed to evaluate the model's ability to reason in real-world scenarios.

### 3.4 Agentic Evaluation in Simulators

Interaction based simulators are needed to evaluate agentic capabilities, instead of relying on static benchmarks.

## References

### Core: Statistical Evaluation Framework

> **[A Statistical Approach to Model Evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals)**
> Miller et al., Anthropic, 2024
>
> The theoretical foundation for Section 2. Key contributions:
> - Treating evaluations as sampling experiments
> - Standard errors and confidence intervals for LLM benchmarks
> - Clustered standard errors for correlated questions
> - Paired difference analysis for model comparison

- Paper: [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (arXiv)

### Evaluation Frameworks

- [LMMs-Eval](https://arxiv.org/html/2407.12772v2) — Base framework
- [FlagEvalMM](https://arxiv.org/html/2506.09081v3) — Multimodal evaluation

### Data Formats

- [Lance Format](https://arxiv.org/html/2504.15247v1) — Columnar storage for ML

### Benchmarks

- [LLM Agent Evaluation Survey](https://arxiv.org/html/2507.21504v1)
- [StreamingBench](https://arxiv.org/html/2411.03628v1) — Video streaming
- [VSI-Bench](https://arxiv.org/html/2505.05456v2) — Spatial intelligence

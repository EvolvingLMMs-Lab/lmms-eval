# Throughput Metrics Documentation

This document describes the inference throughput metrics automatically logged by LMMS-Eval chat models during evaluation.

## Overview

LMMS-Eval chat models automatically log detailed timing metrics during inference to help users understand model performance characteristics. These metrics are logged at the INFO level and provide insights into end-to-end latency, token generation speed, and other performance indicators.

## Metrics Explained

### Core Timing Metrics

- **E2E (End-to-End Latency)**: Total time from request submission to response completion (in seconds)
- **TTFT (Time to First Token)**: Time from request submission until the first token is generated (in seconds)  
- **TPOT (Time Per Output Token)**: Average time to generate each output token after the first (in seconds)
- **Speed (Inference Speed)**: Token generation rate calculated as 1/TPOT (tokens per second)
- **Output Tokens**: Number of tokens generated in the response

### Batch Metrics

For models that process multiple requests in batches:

- **Batch Summary**: Aggregated metrics across all outputs in a batch
- **Total Time**: Total batch processing time
- **Total Tokens**: Sum of all output tokens in the batch
- **Avg Speed**: Average throughput across the entire batch (tokens/s)

## Log Format Examples

### Individual Output Metrics
```
Output 0 - E2E: 2.145s, TTFT: 0.215s, TPOT: 0.048s, Speed: 20.8 tokens/s, Output tokens: 42
```

### Batch Summary Metrics  
```
Batch summary - Total time: 2.145s, Total tokens: 128, Avg speed: 59.7 tokens/s
```

### Single Request Metrics (Non-batched)
```
Inference metrics - E2E: 1.823s, TTFT: 0.182s, TPOT: 0.052s, Speed: 19.2 tokens/s, Output tokens: 32
```

## Supported Models

The following chat models automatically log throughput metrics:

- **sglang_runtime** (`/lmms_eval/models/chat/sglang.py`)
- **vllm_chat** (`/lmms_eval/models/chat/vllm.py`) 
- **llava_hf_chat** (`/lmms_eval/models/chat/llava_hf.py`)
- **openai_compatible_chat** (`/lmms_eval/models/chat/openai_compatible.py`)
- **qwen2_5_vl_chat** (`/lmms_eval/models/chat/qwen2_5_vl.py`)
- **huggingface_chat** (`/lmms_eval/models/chat/huggingface.py`)

## Usage

Throughput metrics are automatically logged during evaluation - no additional configuration is required. To view the metrics:

1. **Command Line Output**: Metrics appear in real-time during evaluation
2. **Log Files**: Metrics are written to log files if logging is configured
3. **Log Level**: Ensure logging level is set to INFO or lower to see metrics

### Example Evaluation Command
```bash
python -m lmms_eval \
    --model sglang_runtime \
    --model_args model=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks mme \
    --batch_size 4 \
    --log_samples \
    --output_path ./results
```

## Metric Calculation Details

### TTFT Calculation
- **Available from model**: Uses actual first token timestamp when provided
- **Estimated**: When unavailable, estimated as 10% of total inference time

### TPOT Calculation  
- **Formula**: `(E2E_latency - TTFT) / (output_tokens - 1)`
- **Single token responses**: Uses full E2E latency as TPOT
- **Batch processing**: Divides total batch time by number of outputs

### Speed Calculation
- **Formula**: `1 / TPOT` (when TPOT > 0)
- **Edge cases**: Set to 0 for single-token responses or zero TPOT

## Performance Analysis

### Interpreting Metrics

- **High TTFT**: May indicate model loading, prompt processing, or scheduling delays
- **High TPOT**: Suggests slower token generation, possibly due to model size or hardware limitations  
- **Low Speed**: Indicates throughput bottlenecks in token generation
- **E2E vs TTFT+TPOT**: Large differences may suggest batching overhead or system delays

### Optimization Insights

- **Reduce TTFT**: Optimize prompt processing, use model caching, improve scheduling
- **Reduce TPOT**: Use faster hardware, optimize model inference, adjust batch sizes
- **Batch Efficiency**: Compare individual vs batch metrics to assess batching benefits

## Troubleshooting

### Missing Metrics
- Ensure model supports throughput logging (see supported models list)
- Check log level is set to INFO or lower
- Verify model implementation includes timing instrumentation

### Inaccurate Metrics  
- TTFT estimates may be imprecise when actual timing unavailable
- Batch metrics average across multiple outputs, individual variance not captured
- Network latency may affect metrics for API-based models

## Implementation Notes

Throughput metrics are implemented consistently across chat models using:
- `time.time()` for wall-clock timing measurements
- Model-specific metadata when available (e.g., SGLang, VLLM native metrics)
- Fallback estimation methods for missing timing data
- Structured logging format for consistent parsing and analysis
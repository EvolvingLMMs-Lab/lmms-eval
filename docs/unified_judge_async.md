# Unified LLM Judge - Async/Await Support and Additional Providers

This document describes the improvements made to the unified LLM judge system, including async/await support for high-throughput scenarios and additional provider support.

## Overview

The enhanced judge system now includes:

1. **Async/Await Support**: High-performance asynchronous evaluation capabilities
2. **Additional Providers**: Support for Anthropic (Claude), Cohere, and Together AI
3. **Concurrent Batch Processing**: Evaluate multiple items concurrently with rate limiting
4. **Automatic Fallback**: Failover to alternative providers when primary fails
5. **Progress Tracking**: Monitor evaluation progress for large datasets

## New Files

- `judge_async.py`: Core async judge implementations
- `judge_utils_async.py`: Async utilities for common evaluation patterns
- `judge_providers.py`: Additional provider implementations (Anthropic, Cohere, Together AI)
- `judge_example_integration.py`: Examples of high-throughput evaluation

## Usage Examples

### Basic Async Evaluation

```python
import asyncio
from lmms_eval.api.judge_utils_async import get_binary_judge_response_async

async def evaluate_single():
    result = await get_binary_judge_response_async(
        question="What is 2+2?",
        answer="4",
        prediction="The answer is 4",
        api_type="openai"
    )
    print(f"Result: {result}")

# Run the async function
asyncio.run(evaluate_single())
```

### Batch Evaluation with Concurrency Control

```python
from lmms_eval.api.judge_utils_async import AsyncSimplifiedJudge

async def evaluate_batch():
    judge = AsyncSimplifiedJudge(
        api_type="openai",
        model_name="gpt-4o-2024-08-06",
        max_concurrent=20  # Process 20 items concurrently
    )
    
    questions = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
    answers = ["4", "6", "8"]
    predictions = ["4", "6", "8"]
    
    results = await judge.evaluate_binary_batch_async(
        questions, answers, predictions
    )
    
    for i, result in enumerate(results):
        print(f"Question {i+1}: {result['result']}")

asyncio.run(evaluate_batch())
```

### Using Alternative Providers

```python
# Anthropic Claude
judge = AsyncSimplifiedJudge(
    api_type="anthropic",
    model_name="claude-3-opus-20240229"
)

# Cohere
judge = AsyncSimplifiedJudge(
    api_type="cohere",
    model_name="command-xlarge"
)

# Together AI
judge = AsyncSimplifiedJudge(
    api_type="together",
    model_name="meta-llama/Llama-2-70b-chat-hf"
)
```

### Evaluation with Automatic Fallback

```python
from lmms_eval.api.judge_async import evaluate_with_fallback, JudgeRequest

async def eval_with_fallback():
    request = JudgeRequest(
        messages=[{"role": "user", "content": "Evaluate if 2+2=4 is correct"}]
    )
    
    # Will try OpenAI first, then fall back to Anthropic and Together AI
    response = await evaluate_with_fallback(
        request,
        primary_api="openai",
        fallback_apis=["anthropic", "together"]
    )
    print(f"Response: {response.content}")
    print(f"Provider used: {response.model_used}")

asyncio.run(eval_with_fallback())
```

### High-Throughput Evaluation

```python
from lmms_eval.api.judge_example_integration import HighThroughputEvaluator

async def evaluate_large_dataset():
    # Create a large dataset
    dataset = [
        {
            "question": f"What is {i} + {i}?",
            "answer": str(2 * i),
            "prediction": str(2 * i)
        }
        for i in range(1000)
    ]
    
    # Initialize evaluator with fallback support
    evaluator = HighThroughputEvaluator(
        api_type="openai",
        max_concurrent=50,  # Process 50 items concurrently
        use_fallback=True
    )
    
    # Run evaluation
    results = await evaluator.evaluate_dataset_async(
        dataset,
        eval_type="binary",
        batch_size=100
    )
    
    # Analyze results
    correct = sum(1 for r in results if r.get("result") == 1)
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

asyncio.run(evaluate_large_dataset())
```

## Environment Variables

Set the following environment variables based on the provider:

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
export API_TYPE="openai"
export MODEL_VERSION="gpt-4o-2024-08-06"
```

### Azure OpenAI
```bash
export AZURE_API_KEY="your-api-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com"
export API_VERSION="2024-02-15-preview"
export API_TYPE="azure"
export MODEL_VERSION="your-deployment-name"
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="your-api-key"
export API_TYPE="anthropic"
export MODEL_VERSION="claude-3-opus-20240229"
```

### Cohere
```bash
export COHERE_API_KEY="your-api-key"
export API_TYPE="cohere"
export MODEL_VERSION="command-xlarge"
```

### Together AI
```bash
export TOGETHER_API_KEY="your-api-key"
export API_TYPE="together"
export MODEL_VERSION="meta-llama/Llama-2-70b-chat-hf"
```

## Performance Comparison

Based on typical usage patterns:

| Method | Items/Second | Notes |
|--------|--------------|-------|
| Synchronous | 1-2 | Sequential processing |
| Async (10 concurrent) | 8-10 | Good for moderate workloads |
| Async (50 concurrent) | 30-40 | Optimal for large datasets |
| Async (100 concurrent) | 40-50 | May hit rate limits |

## Integration with Existing Code

### Converting Synchronous to Async

```python
from lmms_eval.api.judge_example_integration import run_async_evaluation

# Existing synchronous code
def evaluate_sync(dataset):
    results = []
    for item in dataset:
        result = get_binary_judge_response(
            item["question"],
            item["answer"],
            item["prediction"]
        )
        results.append(result)
    return results

# Convert to async
async def evaluate_async(dataset):
    from lmms_eval.api.judge_utils_async import AsyncSimplifiedJudge
    
    judge = AsyncSimplifiedJudge(max_concurrent=20)
    results = await judge.evaluate_binary_batch_async(
        [item["question"] for item in dataset],
        [item["answer"] for item in dataset],
        [item["prediction"] for item in dataset]
    )
    return results

# Run async version in sync code
results = run_async_evaluation(evaluate_async(dataset))
```

## Best Practices

1. **Choose Appropriate Concurrency**: Start with 10-20 concurrent requests and adjust based on rate limits
2. **Use Batching**: Process items in batches to avoid overwhelming the API
3. **Implement Progress Tracking**: For large datasets, add progress callbacks
4. **Handle Rate Limits**: Use exponential backoff and provider fallback
5. **Monitor Costs**: Track token usage across providers

## Error Handling

The async implementation includes:

- Automatic retry with exponential backoff
- Provider fallback when primary fails
- Detailed error logging
- Graceful degradation

## Future Enhancements

Potential future improvements:

1. **Streaming Support**: For real-time evaluation feedback
2. **WebSocket Support**: For persistent connections
3. **Caching Layer**: Redis/memcached integration
4. **Metrics Collection**: Prometheus/Grafana integration
5. **Additional Providers**: Google PaLM, Replicate, etc.
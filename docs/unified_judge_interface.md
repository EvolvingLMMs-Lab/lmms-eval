# Unified Judge Interface Documentation

## Overview

The unified judge interface provides a consistent way to use different LLM/LMM providers (OpenAI, Azure OpenAI, etc.) as judges for evaluation tasks. This interface abstracts away the differences between providers and makes it easy to switch between them.

## Key Components

### 1. JudgeInterface (Abstract Base Class)

The base interface that all judge implementations must follow:

```python
from lmms_eval.api.judge import JudgeInterface, JudgeRequest, JudgeResponse

class MyCustomJudge(JudgeInterface):
    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        # Your implementation here
        pass
    
    def is_available(self) -> bool:
        # Check if service is available
        pass
```

### 2. JudgeConfig

Configuration dataclass for judge models:

```python
from lmms_eval.api.judge import JudgeConfig

config = JudgeConfig(
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=1024,
    top_p=None,
    timeout=60,
    num_retries=5,
    retry_delay=10,
    system_prompt="You are a helpful evaluator",
    response_format="json"  # or "text"
)
```

### 3. JudgeRequest and JudgeResponse

Standard request/response format:

```python
from lmms_eval.api.judge import JudgeRequest, JudgeResponse

# Create a request
request = JudgeRequest(
    messages=[
        {"role": "system", "content": "You are an evaluator"},
        {"role": "user", "content": "Evaluate this response..."}
    ],
    images=["path/to/image.jpg"],  # Optional
    config=config  # Optional, overrides default
)

# Response format
response = JudgeResponse(
    content="The evaluation result...",
    model_used="gpt-4-0613",
    usage={"prompt_tokens": 100, "completion_tokens": 50},
    raw_response={}  # Original API response
)
```

## Quick Start

### Basic Usage

```python
from lmms_eval.api.judge import get_judge

# Create a judge (uses environment variables for API configuration)
judge = get_judge()

# Evaluate
request = JudgeRequest(
    messages=[{"role": "user", "content": "Evaluate this: ..."}]
)
response = judge.evaluate(request)
print(response.content)
```

### With Configuration

```python
from lmms_eval.api.judge import get_judge, JudgeConfig

# Custom configuration
config = JudgeConfig(
    model_name="gpt-4-turbo",
    temperature=0.1,
    max_tokens=2048
)

# Create judge with config
judge = get_judge(api_type="openai", config=config)
```

### Using Helper Functions

```python
from lmms_eval.api.judge_utils import evaluate_with_judge

# Simple evaluation
content, model = evaluate_with_judge(
    prompt="Evaluate this response: ...",
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=1024
)
```

## Migration Guide

### Migrating from Direct API Calls

**Before:**
```python
import os
from openai import OpenAI, AzureOpenAI

API_TYPE = os.getenv("API_TYPE", "openai")
if API_TYPE == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION")
    )

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.0
)
```

**After:**
```python
from lmms_eval.api.judge import get_judge, JudgeRequest

judge = get_judge()  # Automatically uses environment variables
request = JudgeRequest(
    messages=[{"role": "user", "content": "..."}]
)
response = judge.evaluate(request)
```

### Migrating Existing Evaluation Functions

**Before:**
```python
def get_eval(content: str, max_tokens: int):
    headers = {...}
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
```

**After:**
```python
from lmms_eval.api.judge_utils import evaluate_with_judge

def get_eval(content: str, max_tokens: int):
    response, model = evaluate_with_judge(
        prompt=content,
        max_tokens=max_tokens
    )
    return response
```

## Environment Variables

The unified interface uses the same environment variables as before:

- `API_TYPE`: "openai" or "azure" (default: "openai")
- For OpenAI:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_API_URL`: API endpoint (default: https://api.openai.com/v1/chat/completions)
- For Azure:
  - `AZURE_API_KEY`: Your Azure API key
  - `AZURE_ENDPOINT`: Azure endpoint URL
  - `AZURE_API_VERSION`: API version (default: "2024-02-15-preview")

## Advanced Features

### Custom Judge Implementation

```python
from lmms_eval.api.judge import JudgeInterface, JudgeFactory

class CustomJudge(JudgeInterface):
    def __init__(self, config=None):
        super().__init__(config)
        # Your initialization
    
    def evaluate(self, request):
        # Your evaluation logic
        return JudgeResponse(
            content="evaluation result",
            model_used="custom-model"
        )
    
    def is_available(self):
        return True

# Register your judge
JudgeFactory.register_judge("custom", CustomJudge)

# Use it
judge = get_judge(api_type="custom")
```

### Batch Evaluation

```python
from concurrent.futures import ThreadPoolExecutor
from lmms_eval.api.judge import get_judge, JudgeRequest

judge = get_judge()
requests = [
    JudgeRequest(messages=[{"role": "user", "content": f"Evaluate {i}"}])
    for i in range(10)
]

with ThreadPoolExecutor(max_workers=5) as executor:
    responses = list(executor.map(judge.evaluate, requests))
```

### Image Evaluation

```python
from lmms_eval.api.judge import get_judge, JudgeRequest

judge = get_judge(model_name="gpt-4-vision-preview")
request = JudgeRequest(
    messages=[{"role": "user", "content": "What's in this image?"}],
    images=["path/to/image.jpg", "path/to/another.png"]
)
response = judge.evaluate(request)
```

## Best Practices

1. **Error Handling**: The interface includes automatic retry logic. Configure retries in JudgeConfig.

2. **Model Selection**: Always specify the model explicitly for reproducibility:
   ```python
   config = JudgeConfig(model_name="gpt-4-0613")
   ```

3. **Response Parsing**: Use provided utility functions:
   ```python
   from lmms_eval.api.judge_utils import parse_score, extract_json_score
   
   scores = parse_score(response.content)  # For "8.5 9.0" format
   json_data = extract_json_score(response.content)  # For JSON responses
   ```

4. **Logging**: The interface uses loguru for consistent logging:
   ```python
   from loguru import logger as eval_logger
   eval_logger.info("Starting evaluation...")
   ```

## Backward Compatibility

The interface maintains backward compatibility through:

1. `get_eval()` function in judge_utils
2. `LegacyJudgeAdapter` class for gradual migration
3. Same environment variable names

This allows gradual migration without breaking existing code.

## Examples

See the following files for complete examples:
- `/lmms_eval/tasks/llava-bench-coco/utils_unified.py` - Refactored evaluation using unified interface
- `/lmms_eval/api/judge.py` - Core interface implementation
- `/lmms_eval/api/judge_utils.py` - Helper utilities and backward compatibility
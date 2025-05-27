# Unified Judge Interface

The unified judge interface provides a standardized way to use LLM-based evaluation across different tasks in lmms-eval. It supports multiple evaluation types and API providers while maintaining backward compatibility with existing code.

## Features

- **Multiple Judge Types**: Binary (correct/incorrect), Comparative (scoring two responses), Score-based, and Custom rubric evaluation
- **API Provider Support**: OpenAI and Azure OpenAI with easy extensibility for other providers
- **Multimodal Support**: Handles both text-only and image+text evaluations
- **Backward Compatibility**: Provides adapter functions for existing code
- **Flexible Configuration**: Environment variables or explicit configuration
- **Robust Error Handling**: Automatic retries with exponential backoff

## Architecture Overview

The unified judge API consists of three main layers:

1. **Core Judge Interface** (`lmms_eval.api.judge`): Abstract base classes and factory pattern for different API providers
2. **Judge Utilities** (`lmms_eval.api.judge_utils`): Helper classes for specific evaluation types and backward compatibility
3. **Task Integration**: Refactored task utilities that use the unified API

## Quick Start

### Basic Usage

```python
from lmms_eval.api.judge_utils import SimplifiedJudge, get_binary_judge_response

# Quick evaluation
result = get_binary_judge_response(
    question="What is 2+2?",
    answer="4",
    prediction="The answer is 4",
    output_format="0/1"  # or "yes/no"
)
print(result)  # Output: 1

# Using the SimplifiedJudge class
judge = SimplifiedJudge(model_name="gpt-4o-2024-08-06")

# Binary evaluation
result = judge.evaluate_binary(
    question="What is the capital of France?",
    answer="Paris",
    prediction="The capital of France is Paris.",
    output_format="0/1"
)
print(result["result"])  # Output: 1

# Comparative evaluation
result = judge.evaluate_comparative(
    question="Explain photosynthesis",
    response1="Photosynthesis is the process by which plants make food...",
    response2="Plants eat sunlight to grow",
    score_range=(1, 10)
)
print(result["scores"])  # Output: (8.5, 3.0)
```

### Advanced Usage

```python
from lmms_eval.api.judge import JudgeConfig, JudgeFactory, JudgeRequest

# Custom configuration
config = JudgeConfig(
    model_name="gpt-4o-2024-08-06",
    temperature=0.0,
    max_tokens=256,
    num_retries=3,
    judge_type="binary"
)

# Create judge instance
judge = JudgeFactory.create_judge(api_type="openai", config=config)

# Create request with images
request = JudgeRequest(
    messages=[{"role": "user", "content": "Is this a cat?"}],
    images=["path/to/image.jpg"],
    config=config
)

# Evaluate
response = judge.evaluate(request)
print(response.content)
```

## Configuration

### Environment Variables

```bash
# API Type (openai or azure)
export API_TYPE=openai

# OpenAI Configuration
export OPENAI_API_KEY=your-api-key
export OPENAI_API_URL=https://api.openai.com/v1/chat/completions
export MODEL_VERSION=gpt-4o-2024-08-06

# Azure OpenAI Configuration
export AZURE_API_KEY=your-api-key
export AZURE_ENDPOINT=your-endpoint
export AZURE_API_VERSION=2024-02-15-preview
```

### Programmatic Configuration

```python
from lmms_eval.api.judge import JudgeConfig

config = JudgeConfig(
    model_name="gpt-4o-2024-08-06",
    temperature=0.0,
    max_tokens=1024,
    timeout=60,
    num_retries=5,
    retry_delay=10,
    judge_type="comparative",
    score_range=(1, 10)
)
```

## Judge Types and Use Cases

### 1. Binary Judge
For evaluating correctness of answers (correct/incorrect, yes/no).

```python
from lmms_eval.api.judge_utils import SimplifiedJudge

judge = SimplifiedJudge()

# Built-in prompt template
result = judge.evaluate_binary(
    question="What is 2+2?",
    answer="4",
    prediction="4",
    output_format="0/1"  # or "yes/no"
)

# Custom prompt
custom_prompt = "Compare {prediction} with {answer}. Return 1 if correct, 0 if wrong."
result = judge.evaluate_binary(
    question="What is 2+2?",
    answer="4",
    prediction="4",
    custom_prompt=custom_prompt
)
```

### 2. Comparative Judge
For comparing and scoring two responses.

```python
result = judge.evaluate_comparative(
    question="Explain gravity",
    response1="Gravity is a fundamental force...",
    response2="Things fall down",
    context="This is a physics question",
    score_range=(1, 10)
)
scores = result["scores"]  # (score1, score2)
```

### 3. Custom Rubric Evaluation
For evaluating against multiple criteria.

```python
rubric = {
    "accuracy": "Is the information factually correct?",
    "completeness": "Does it cover all key points?",
    "clarity": "Is it easy to understand?"
}

result = judge.evaluate_with_rubric(
    question="Explain the water cycle",
    prediction="Water evaporates, forms clouds, and rains down...",
    rubric=rubric
)
print(result["scores"])  # {"accuracy": 8, "completeness": 7, "clarity": 9}
```

## Prompt Templates

The unified judge interface provides optimized prompt templates for different evaluation scenarios:

### Binary Evaluation Template
Used for correct/incorrect assessments:
- Handles multiple-choice questions (checks if prediction matches answer)
- Handles open-ended questions (checks semantic equivalence)
- Ignores formatting differences
- Clear output format specification

### Comparative Evaluation Template
Used for comparing two responses:
- Rates on a configurable scale (default 1-10)
- Considers helpfulness, relevance, accuracy, and detail
- Outputs scores on first line for easy parsing
- Provides detailed explanation

### Correctness Template
Specialized for math/science problems:
- Focuses on mathematical/semantic correctness only
- Ignores LaTeX formatting, symbols, and wrappers
- Disregards solution process, only checks final answer
- Binary yes/no output

## Migration Guide

### From Direct API Calls

**Before:**
```python
def get_chat_response(content, max_tokens, retries=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    
    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    
    # Complex retry logic
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Error handling
            pass
```

**After:**
```python
from lmms_eval.api.judge_utils import SimplifiedJudge

judge = SimplifiedJudge()
result = judge.evaluate_binary(
    question=question,
    answer=answer,
    prediction=prediction
)
score = result["result"]
```

### From Task-Specific Implementations

**MMMU Task (Before):**
```python
llm_judge_prompt = JUDGE_RULES.format(question=formatted_question, answer=answer, pred=pred)
llm_judge_score = get_chat_response(llm_judge_prompt, max_tokens=20, retries=3)
```

**MMMU Task (After):**
```python
result = mmmu_judge.evaluate_binary(
    question=formatted_question,
    answer=answer,
    prediction=pred,
    output_format="0/1"
)
scores.append(result["raw_response"])
```

**LLaVA-Bench Task (Before):**
```python
review, model_name = get_eval(content, 1024)
scores = parse_score(review)
```

**LLaVA-Bench Task (After):**
```python
result = llava_judge.evaluate_comparative(
    question=question,
    response1=ans1,
    response2=ans2,
    context=context,
    custom_prompt=content,
    score_range=(1, 10)
)
scores = list(result["scores"])
```

## Backward Compatibility

The unified judge interface maintains full backward compatibility through:

1. **Legacy Function Support**:
   ```python
   # These functions still work
   from lmms_eval.api.judge_utils import get_eval, parse_score, get_chat_response
   
   content, model = get_eval(prompt, max_tokens=1024)
   scores = parse_score(content)
   response = get_chat_response(prompt, max_tokens=256)
   ```

2. **Environment Variable Compatibility**: Uses the same environment variables as before

3. **Response Format Preservation**: Maintains the same output formats expected by existing code

## Extending the Judge Interface

### Adding a New API Provider

```python
from lmms_eval.api.judge import JudgeInterface, JudgeFactory

class AnthropicJudge(JudgeInterface):
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
    def is_available(self):
        return bool(self.api_key)
        
    def evaluate(self, request):
        # Implement Anthropic API call
        # Return JudgeResponse
        pass

# Register the new judge
JudgeFactory.register_judge("anthropic", AnthropicJudge)

# Use it
judge = JudgeFactory.create_judge(api_type="anthropic")
```

### Creating Custom Judge Types

```python
from lmms_eval.api.judge_utils import SimplifiedJudge, JudgePromptBuilder

class DomainSpecificJudge(SimplifiedJudge):
    def evaluate_medical_accuracy(self, question, answer, prediction, **kwargs):
        # Build specialized prompt for medical domain
        prompt = f"""
        As a medical expert, evaluate if the following diagnosis is correct.
        
        Patient symptoms: {question}
        Correct diagnosis: {answer}
        Predicted diagnosis: {prediction}
        
        Consider medical accuracy, completeness, and safety.
        Output: Correct/Incorrect/Partially Correct
        """
        
        request = JudgeRequest(
            messages=[{"role": "user", "content": prompt}],
            config=self.config
        )
        
        response = self.judge.evaluate(request)
        
        # Custom parsing logic
        if "correct" in response.content.lower():
            if "partially" in response.content.lower():
                return 0.5
            return 1.0
        return 0.0
```

## Best Practices

1. **Model Selection**: Always specify model explicitly for reproducibility
   ```python
   judge = SimplifiedJudge(model_name="gpt-4o-2024-08-06")
   ```

2. **Error Handling**: Check success status and handle failures
   ```python
   result = judge.evaluate_binary(...)
   if result["success"]:
       score = result["result"]
   else:
       # Handle failure
       logger.error(f"Evaluation failed: {result.get('error_message')}")
   ```

3. **Token Optimization**: Set appropriate max_tokens
   ```python
   judge = SimplifiedJudge(max_tokens=256)  # For simple yes/no
   judge = SimplifiedJudge(max_tokens=1024)  # For detailed explanations
   ```

4. **Prompt Engineering**: Use custom prompts for domain-specific needs
   ```python
   domain_prompt = "Evaluate as a {domain} expert: ..."
   result = judge.evaluate_binary(..., custom_prompt=domain_prompt)
   ```

5. **Batch Processing**: Process multiple items efficiently
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def evaluate_item(item):
       return judge.evaluate_binary(**item)
   
   with ThreadPoolExecutor(max_workers=5) as executor:
       results = list(executor.map(evaluate_item, items))
   ```

## Performance Considerations

- **Caching**: Consider caching judge responses for identical inputs
- **Rate Limiting**: Configure retry_delay to respect API rate limits
- **Parallel Processing**: Use thread pools for batch evaluation
- **Token Usage**: Monitor token usage through response.usage field

## Troubleshooting

### Common Issues

1. **Empty Responses**
   - Check API key configuration
   - Verify endpoint URLs
   - Check model availability

2. **Parsing Errors**
   - Ensure output_format matches expected format
   - Check for prompt template issues
   - Verify response parser logic

3. **Rate Limit Errors**
   - Increase retry_delay in configuration
   - Implement exponential backoff
   - Use batch processing with delays

4. **Timeout Errors**
   - Increase timeout value
   - Reduce max_tokens for faster responses
   - Check network connectivity

### Debug Mode

```python
from loguru import logger

# Enable debug logging
logger.add("judge_debug.log", level="DEBUG")

# Inspect full results
result = judge.evaluate_binary(...)
print(f"Success: {result['success']}")
print(f"Raw response: {result['raw_response']}")
print(f"Prompt used: {result['prompt']}")
print(f"Model: {result['model']}")
```

## Real-World Examples

The following tasks have been refactored to use the unified judge API:

1. **MMMU** (`lmms_eval/tasks/mmmu/utils.py`): Binary evaluation for academic questions
2. **LLaVA-in-the-Wild** (`lmms_eval/tasks/llava-in-the-wild/utils.py`): Comparative evaluation with custom rules
3. **LLaVA-Wilder** (`lmms_eval/tasks/llava_wilder/utils.py`): Multimodal comparative evaluation
4. **K12** (`lmms_eval/tasks/k12/utils.py`): Correctness evaluation for educational content

Each demonstrates different aspects of the unified API and can serve as templates for new implementations.

## Future Enhancements

- Async/await support for high-throughput scenarios
- Additional provider support (Anthropic, Cohere, etc.)
- Built-in response caching
- Structured output parsing (JSON mode)
- Multi-turn conversation support
- Streaming response support
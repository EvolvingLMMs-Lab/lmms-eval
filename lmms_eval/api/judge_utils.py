"""
Judge utilities for specific evaluation types
"""

import re
import os
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

from loguru import logger as eval_logger
from .judge import JudgeRequest, JudgeResponse, JudgeConfig, JudgeFactory


# Default prompts for different judge types
BINARY_JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output {positive} for fully correct answers and {negative} for any other case.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score {positive} if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score {positive} if the prediction matches the answer semantically, it can be in different format.
  * Score {negative} for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
{positive} or {negative}"""


COMPARATIVE_JUDGE_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of {min_score} to {max_score}, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Question]
{question}

{context_section}

[Assistant 1]
{response1}
[End of Assistant 1]

[Assistant 2]
{response2}
[End of Assistant 2]

[System]
{evaluation_instruction}"""


CORRECTNESS_JUDGE_PROMPT = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "{positive}" if the solution is correct or "{negative}" if it is incorrect.
Only return "{positive}" or "{negative}" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{prediction}
--------------------------------"""


class JudgePromptBuilder:
    """Helper class to build prompts for different judge types"""
    
    @staticmethod
    def build_binary_prompt(
        question: str,
        answer: str,
        prediction: str,
        output_format: str = "0/1",
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build prompt for binary evaluation"""
        if custom_prompt:
            return custom_prompt.format(
                question=question,
                answer=answer,
                pred=prediction,
                prediction=prediction,
                **kwargs
            )
        
        positive, negative = ("1", "0") if output_format == "0/1" else ("Yes", "No")
        
        return BINARY_JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            prediction=prediction,
            positive=positive,
            negative=negative
        )
    
    @staticmethod
    def build_comparative_prompt(
        question: str,
        response1: str,
        response2: str,
        context: Optional[str] = None,
        score_range: Tuple[int, int] = (1, 10),
        custom_prompt: Optional[str] = None,
        evaluation_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build prompt for comparative evaluation"""
        if custom_prompt:
            return custom_prompt.format(
                question=question,
                response1=response1,
                response2=response2,
                context=context or "",
                **kwargs
            )
        
        context_section = f"[Context]\n{context}\n\n" if context else ""
        
        if not evaluation_instruction:
            evaluation_instruction = f"Please provide scores from {score_range[0]} to {score_range[1]}."
        
        return COMPARATIVE_JUDGE_PROMPT.format(
            question=question,
            response1=response1,
            response2=response2,
            context_section=context_section,
            min_score=score_range[0],
            max_score=score_range[1],
            evaluation_instruction=evaluation_instruction
        )
    
    @staticmethod
    def build_correctness_prompt(
        question: str,
        answer: str,
        prediction: str,
        output_format: str = "yes/no",
        **kwargs
    ) -> str:
        """Build prompt for correctness evaluation"""
        positive, negative = ("Yes", "No") if output_format == "yes/no" else ("1", "0")
        
        return CORRECTNESS_JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            prediction=prediction,
            positive=positive,
            negative=negative
        )


class ResponseParser:
    """Helper class to parse different types of judge responses"""
    
    @staticmethod
    def parse_binary_response(response: str, output_format: str = "0/1") -> Union[int, bool]:
        """Parse binary response (0/1 or yes/no)"""
        response = response.strip().lower()
        
        if output_format == "0/1":
            # Check for various formats of 1
            if any(pattern in response for pattern in ["1", "[1]", "score: 1", "answer: 1"]):
                return 1
            else:
                return 0
        else:
            # yes/no format
            return response == "yes" or response.startswith("yes")
    
    @staticmethod
    def parse_score_response(response: str, score_range: Optional[Tuple[float, float]] = None) -> float:
        """Parse a single score from response"""
        try:
            # Try to extract first number from response
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range if provided
                if score_range:
                    score = max(score_range[0], min(score, score_range[1]))
                return score
        except Exception as e:
            pass
        
        # Return minimum score as default
        return score_range[0] if score_range else 0.0
    
    @staticmethod
    def parse_comparative_response(response: str) -> Tuple[float, float]:
        """Parse comparative scores from response"""
        try:
            # Extract scores from first line
            lines = response.strip().split('\n')
            if lines:
                score_line = lines[0]
                # Handle different separators
                score_line = score_line.replace(',', ' ').replace(';', ' ')
                scores = re.findall(r'-?\d+(?:\.\d+)?', score_line)
                
                if len(scores) >= 2:
                    return float(scores[0]), float(scores[1])
        except Exception as e:
            pass
        
        return -1.0, -1.0
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """Parse JSON response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except Exception as e:
            pass
        
        return {}


class SimplifiedJudge:
    """Simplified interface for common judge operations"""
    
    def __init__(self, model_name: Optional[str] = None, api_type: Optional[str] = None, 
                 api_key: Optional[str] = None, azure_endpoint: Optional[str] = None, 
                 api_version: Optional[str] = None, **config_kwargs):
        """Initialize judge with optional configuration
        
        Args:
            model_name: Model name to use (defaults to MODEL_VERSION env var)
            api_type: API type ('openai' or 'azure', defaults to API_TYPE env var)
            api_key: API key (defaults to OPENAI_API_KEY or AZURE_API_KEY env var)
            azure_endpoint: Azure endpoint (only for Azure, defaults to AZURE_ENDPOINT env var)
            api_version: API version (only for Azure, defaults to API_VERSION env var)
            **config_kwargs: Additional configuration parameters
        """
        # Get defaults from environment
        if api_type is None:
            api_type = os.getenv("API_TYPE", "openai")
            
        # Set environment variables if provided (for backward compatibility)
        if api_type == "azure":
            if azure_endpoint:
                os.environ["AZURE_ENDPOINT"] = azure_endpoint
            if api_key:
                os.environ["AZURE_API_KEY"] = api_key
            if api_version:
                os.environ["API_VERSION"] = api_version
        elif api_type == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                
        self.config = JudgeConfig(
            model_name=model_name or os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06"),
            **config_kwargs
        )
        self.judge = JudgeFactory.create_judge(api_type=api_type, config=self.config)
    
    def evaluate_binary(
        self,
        question: str,
        answer: str,
        prediction: str,
        output_format: str = "0/1",
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate binary correctness"""
        # Build prompt
        prompt = JudgePromptBuilder.build_binary_prompt(
            question=question,
            answer=answer,
            prediction=prediction,
            output_format=output_format,
            custom_prompt=custom_prompt,
            **kwargs
        )
        
        # Create request
        request = JudgeRequest(
            messages=[{"role": "user", "content": prompt}],
            question=question,
            answer=answer,
            prediction=prediction,
            config=self.config
        )
        
        # Evaluate
        response = self.judge.evaluate(request)
        
        # Parse result
        parsed_result = ResponseParser.parse_binary_response(response.content, output_format)
        
        return {
            "result": parsed_result,
            "raw_response": response.content,
            "model": response.model_used,
            "prompt": prompt,
            "success": response.success
        }
    
    def evaluate_comparative(
        self,
        question: str,
        response1: str,
        response2: str,
        context: Optional[str] = None,
        score_range: Tuple[int, int] = (1, 10),
        custom_prompt: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate comparative responses"""
        # Build prompt
        prompt = JudgePromptBuilder.build_comparative_prompt(
            question=question,
            response1=response1,
            response2=response2,
            context=context,
            score_range=score_range,
            custom_prompt=custom_prompt,
            **kwargs
        )
        
        # Create request
        request = JudgeRequest(
            messages=[{"role": "user", "content": prompt}],
            question=question,
            response1=response1,
            response2=response2,
            context=context,
            images=images,
            config=self.config
        )
        
        # Evaluate
        response = self.judge.evaluate(request)
        
        # Parse result
        scores = ResponseParser.parse_comparative_response(response.content)
        
        return {
            "scores": scores,
            "raw_response": response.content,
            "model": response.model_used,
            "prompt": prompt,
            "success": response.success
        }
    
    def evaluate_with_rubric(
        self,
        question: str,
        prediction: str,
        rubric: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate with a custom rubric"""
        # Build rubric prompt
        rubric_text = "\n".join([f"- {k}: {v}" for k, v in rubric.items()])
        
        prompt = f"""Evaluate the following response according to the given rubric.

Question: {question}

Response: {prediction}

Rubric:
{rubric_text}

Provide a JSON response with scores for each rubric item."""
        
        # Create request with JSON response format
        config = JudgeConfig(
            model_name=self.config.model_name,
            response_format="json",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        request = JudgeRequest(
            messages=[{"role": "user", "content": prompt}],
            question=question,
            prediction=prediction,
            config=config
        )
        
        # Evaluate
        response = self.judge.evaluate(request)
        
        # Parse JSON result
        parsed_result = ResponseParser.parse_json_response(response.content)
        
        return {
            "scores": parsed_result,
            "raw_response": response.content,
            "model": response.model_used,
            "prompt": prompt,
            "success": response.success
        }


# Convenience functions for backward compatibility
def get_binary_judge_response(
    question: str,
    answer: str, 
    prediction: str,
    model_name: Optional[str] = None,
    output_format: str = "0/1",
    **kwargs
) -> Union[int, bool]:
    """Quick function to get binary judge response"""
    judge = SimplifiedJudge(model_name=model_name)
    result = judge.evaluate_binary(question, answer, prediction, output_format, **kwargs)
    return result["result"]


def get_comparative_scores(
    question: str,
    response1: str,
    response2: str,
    model_name: Optional[str] = None,
    **kwargs
) -> Tuple[float, float]:
    """Quick function to get comparative scores"""
    judge = SimplifiedJudge(model_name=model_name)
    result = judge.evaluate_comparative(question, response1, response2, **kwargs)
    return result["scores"]


# Legacy compatibility functions
def parse_score(review: str) -> List[float]:
    """
    Parse scores from judge review text
    Compatible with existing llava evaluation format
    """
    scores = ResponseParser.parse_comparative_response(review)
    return list(scores)


def get_eval(content: str, max_tokens: int = 1024, retries: int = 5) -> Tuple[str, str]:
    """
    Backward compatible function for existing get_eval calls
    """
    config = JudgeConfig(
        model_name=os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06"),
        max_tokens=max_tokens,
        num_retries=retries,
        temperature=0.2
    )
    
    judge = JudgeFactory.create_judge(config=config)
    
    request = JudgeRequest(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer."
            },
            {"role": "user", "content": content}
        ]
    )
    
    try:
        response = judge.evaluate(request)
        return response.content, response.model_used
    except Exception as e:
        eval_logger.error(f"Evaluation failed: {e}")
        return "", ""


def get_chat_response(content: str, max_tokens: int = 256, retries: int = 5) -> str:
    """
    Backward compatible function for k12 style evaluation
    """
    response, _ = get_eval(content, max_tokens, retries)
    return response
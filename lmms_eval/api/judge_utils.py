"""
Utility functions for judge evaluation
Provides helper functions and backward compatibility for existing code
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Union
from loguru import logger as eval_logger

from .judge import (
    JudgeInterface,
    JudgeConfig, 
    JudgeRequest,
    JudgeResponse,
    get_judge,
)


def create_judge_request(
    prompt: str,
    system_prompt: Optional[str] = None,
    images: Optional[List[Union[str, bytes]]] = None,
    response_format: Optional[str] = None,
) -> JudgeRequest:
    """
    Create a JudgeRequest from simple parameters
    
    Args:
        prompt: The user prompt for evaluation
        system_prompt: Optional system prompt
        images: Optional list of images (paths or base64)
        response_format: 'json' or 'text'
        
    Returns:
        JudgeRequest instance
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    config = None
    if response_format:
        config = JudgeConfig(
            model_name="gpt-4",  # Will be overridden by actual config
            response_format=response_format
        )
    
    return JudgeRequest(
        messages=messages,
        images=images,
        config=config
    )


def parse_score(review: str) -> List[float]:
    """
    Parse scores from judge review text
    Compatible with existing llava evaluation format
    
    Args:
        review: Judge review text containing scores
        
    Returns:
        List of parsed scores, or [-1, -1] if parsing fails
    """
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            eval_logger.debug(f"Error parsing score: {review}")
            return [-1, -1]
    except Exception as e:
        eval_logger.debug(f"Exception parsing score: {e}")
        eval_logger.debug(f"Error review: {review}")
        return [-1, -1]


def extract_json_score(response: str) -> Dict[str, Any]:
    """
    Extract JSON score from judge response
    
    Args:
        response: Judge response that may contain JSON
        
    Returns:
        Extracted JSON dict or empty dict if parsing fails
    """
    import json
    
    # Try to parse the entire response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON object
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    eval_logger.warning(f"Could not extract JSON from response: {response[:200]}...")
    return {}


def evaluate_with_judge(
    prompt: str,
    model_name: str = "gpt-4",
    system_prompt: Optional[str] = None,
    images: Optional[List[Union[str, bytes]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: Optional[str] = None,
    api_type: Optional[str] = None,
    num_retries: int = 3,
) -> Tuple[str, str]:
    """
    Simple evaluation function that returns (content, model_used)
    Backward compatible with existing code
    
    Args:
        prompt: Evaluation prompt
        model_name: Model to use for evaluation
        system_prompt: Optional system prompt
        images: Optional images for evaluation
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        response_format: 'json' or 'text'
        api_type: 'openai' or 'azure' (defaults to environment)
        num_retries: Number of retries on failure
        
    Returns:
        Tuple of (response_content, model_used)
    """
    try:
        # Create configuration
        config = JudgeConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            response_format=response_format,
            num_retries=num_retries,
        )
        
        # Get judge instance
        judge = get_judge(api_type=api_type, config=config)
        
        # Create request
        request = create_judge_request(
            prompt=prompt,
            images=images,
        )
        
        # Evaluate
        response = judge.evaluate(request)
        
        return response.content, response.model_used
        
    except Exception as e:
        eval_logger.error(f"Judge evaluation failed: {e}")
        return "", ""


# Backward compatibility functions
def get_eval(
    content: str,
    max_tokens: int = 1024,
    retries: int = 3,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Tuple[str, str]:
    """
    Backward compatible function for existing get_eval calls
    
    Args:
        content: The prompt content
        max_tokens: Maximum tokens to generate
        retries: Number of retries
        model_name: Model name (optional)
        temperature: Temperature for generation
        
    Returns:
        Tuple of (response_content, model_used)
    """
    return evaluate_with_judge(
        prompt=content,
        model_name=model_name or "gpt-4",
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=retries,
    )


class LegacyJudgeAdapter:
    """
    Adapter class for migrating existing judge implementations
    Provides backward compatibility while using the new unified interface
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_type: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_type = api_type
        self.config = JudgeConfig(model_name=model_name, **kwargs)
        self.judge = get_judge(api_type=api_type, config=self.config)
    
    def evaluate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Evaluate using the unified interface"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request = JudgeRequest(messages=messages, images=images)
        response = self.judge.evaluate(request)
        
        return response.content, response.model_used
    
    def query(self, *args, **kwargs):
        """Backward compatibility for query method"""
        # Convert old query format to new evaluate format
        if args and isinstance(args[0], str):
            return self.evaluate(args[0], **kwargs)
        else:
            raise ValueError("Unsupported query format")
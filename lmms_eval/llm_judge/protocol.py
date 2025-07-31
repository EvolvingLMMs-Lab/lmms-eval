from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Configuration for retry logic
DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_DELAY = 10  # seconds


@dataclass
class ServerConfig:
    """Configuration for judge models"""

    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    timeout: int = 60
    num_retries: int = DEFAULT_NUM_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    max_concurrent: int = 10  # Maximum concurrent requests

    # Additional config for specific judge tasks
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # 'json' or 'text'

    # Judge-specific parameters
    judge_type: str = "general"  # 'general', 'binary', 'score', 'comparative'
    output_format: Optional[str] = None  # For binary: '0/1' or 'yes/no'
    score_range: Optional[Tuple[float, float]] = None  # For scoring judges
    evaluation_criteria: Optional[Dict[str, Any]] = None  # Custom evaluation criteria


@dataclass
class Request:
    """Standard request format for judge evaluation"""

    messages: List[Dict[str, Any]]
    images: Optional[List[Union[str, bytes]]] = None  # Image paths or base64 encoded
    config: Optional[ServerConfig] = None

    # Structured input for specific judge types
    question: Optional[str] = None
    answer: Optional[str] = None  # Ground truth
    prediction: Optional[str] = None  # Model prediction
    context: Optional[str] = None  # Additional context
    options: Optional[List[str]] = None  # For multiple choice

    # For comparative evaluation
    response1: Optional[str] = None
    response2: Optional[str] = None

    # Custom evaluation prompt
    custom_prompt: Optional[str] = None
    prompt_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Standard response format from judge evaluation"""

    content: str
    model_used: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None

    # Parsed results for specific judge types
    parsed_result: Optional[Union[int, float, bool, Tuple[float, float], Dict[str, Any]]] = None
    success: bool = True
    error_message: Optional[str] = None

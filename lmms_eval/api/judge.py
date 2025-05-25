"""
Unified Judge Interface for LLM/LMM evaluation
This module provides a unified interface for using different LLM providers as judges.
"""

import abc
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import time
import requests
import json
from loguru import logger as eval_logger

# Configuration for retry logic
DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_DELAY = 10  # seconds


@dataclass
class JudgeConfig:
    """Configuration for judge models"""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: Optional[float] = None
    timeout: int = 60
    num_retries: int = DEFAULT_NUM_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    
    # Additional config for specific judge tasks
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # 'json' or 'text'


@dataclass 
class JudgeRequest:
    """Standard request format for judge evaluation"""
    messages: List[Dict[str, Any]]
    images: Optional[List[Union[str, bytes]]] = None  # Image paths or base64 encoded
    config: Optional[JudgeConfig] = None


@dataclass
class JudgeResponse:
    """Standard response format from judge evaluation"""
    content: str
    model_used: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class JudgeInterface(abc.ABC):
    """Abstract base class for judge implementations"""
    
    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig(model_name="gpt-4")
        
    @abc.abstractmethod
    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """
        Evaluate the given request and return a response
        
        Args:
            request: JudgeRequest containing the evaluation context
            
        Returns:
            JudgeResponse with the evaluation result
        """
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the judge service is available"""
        pass
    
    def prepare_messages(self, request: JudgeRequest) -> List[Dict[str, Any]]:
        """Prepare messages in the format expected by the API"""
        messages = request.messages.copy()
        
        # Add system prompt if configured
        if self.config.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
            
        return messages


class OpenAIJudge(JudgeInterface):
    """OpenAI API implementation of the Judge interface"""
    
    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.use_client = True
        except ImportError:
            eval_logger.warning("OpenAI client not available, falling back to requests")
            self.use_client = False
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using OpenAI API"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        config = request.config or self.config
        messages = self.prepare_messages(request)
        
        # Handle images if present
        if request.images:
            messages = self._add_images_to_messages(messages, request.images)
        
        # Prepare payload
        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if config.top_p is not None:
            payload["top_p"] = config.top_p
            
        if config.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        # Make API call with retries
        for attempt in range(config.num_retries):
            try:
                if self.use_client:
                    response = self.client.chat.completions.create(**payload)
                    content = response.choices[0].message.content
                    model_used = response.model
                    usage = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else None
                    raw_response = response
                else:
                    response = self._make_request(payload, config.timeout)
                    content = response["choices"][0]["message"]["content"]
                    model_used = response["model"]
                    usage = response.get("usage")
                    raw_response = response
                
                return JudgeResponse(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    raw_response=raw_response
                )
                
            except Exception as e:
                eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise
    
    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """Make HTTP request to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to the last user message"""
        import base64
        from PIL import Image
        from io import BytesIO
        
        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                # Convert content to list format if needed
                if isinstance(messages[i]["content"], str):
                    messages[i]["content"] = [{"type": "text", "text": messages[i]["content"]}]
                
                # Add images
                for image in images:
                    if isinstance(image, str):
                        # File path
                        base64_image = self._encode_image(image)
                        messages[i]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
                    elif isinstance(image, bytes):
                        # Already base64 encoded
                        messages[i]["content"].append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{image.decode()}"}
                        })
                break
                
        return messages
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        import base64
        from PIL import Image
        from io import BytesIO
        
        img = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class AzureOpenAIJudge(JudgeInterface):
    """Azure OpenAI implementation of the Judge interface"""
    
    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("AZURE_API_KEY", "")
        self.api_endpoint = os.getenv("AZURE_ENDPOINT", "")
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
        
        # Initialize Azure OpenAI client
        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.api_endpoint,
                api_version=self.api_version
            )
            self.use_client = True
        except ImportError:
            eval_logger.warning("Azure OpenAI client not available, falling back to requests")
            self.use_client = False
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_endpoint)
    
    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Azure OpenAI API"""
        if not self.is_available():
            raise ValueError("Azure OpenAI API credentials not configured")
        
        config = request.config or self.config
        messages = self.prepare_messages(request)
        
        # Handle images if present
        if request.images:
            messages = self._add_images_to_messages(messages, request.images)
        
        # Prepare payload
        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if config.top_p is not None:
            payload["top_p"] = config.top_p
            
        if config.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        # Make API call with retries
        for attempt in range(config.num_retries):
            try:
                if self.use_client:
                    response = self.client.chat.completions.create(**payload)
                    content = response.choices[0].message.content
                    model_used = response.model
                    usage = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else None
                    raw_response = response
                else:
                    response = self._make_request(payload, config.timeout)
                    content = response["choices"][0]["message"]["content"]
                    model_used = response["model"]
                    usage = response.get("usage")
                    raw_response = response
                
                return JudgeResponse(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    raw_response=raw_response
                )
                
            except Exception as e:
                eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise
    
    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """Make HTTP request to Azure OpenAI API"""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        # Construct the full URL
        deployment_name = payload["model"]
        url = f"{self.api_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to messages - same as OpenAI implementation"""
        # Azure OpenAI uses the same format as OpenAI
        return OpenAIJudge._add_images_to_messages(self, messages, images)
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 - same as OpenAI implementation"""
        return OpenAIJudge._encode_image(self, image_path)


class JudgeFactory:
    """Factory for creating judge instances based on configuration"""
    
    _judge_classes = {
        "openai": OpenAIJudge,
        "azure": AzureOpenAIJudge,
    }
    
    @classmethod
    def create_judge(
        cls,
        api_type: Optional[str] = None,
        config: Optional[JudgeConfig] = None
    ) -> JudgeInterface:
        """
        Create a judge instance based on API type
        
        Args:
            api_type: Type of API to use ('openai' or 'azure'). 
                     If None, will use API_TYPE environment variable
            config: Configuration for the judge
            
        Returns:
            JudgeInterface instance
        """
        if api_type is None:
            api_type = os.getenv("API_TYPE", "openai").lower()
        
        if api_type not in cls._judge_classes:
            raise ValueError(f"Unknown API type: {api_type}. Supported types: {list(cls._judge_classes.keys())}")
        
        judge_class = cls._judge_classes[api_type]
        return judge_class(config=config)
    
    @classmethod
    def register_judge(cls, api_type: str, judge_class: type):
        """Register a new judge implementation"""
        if not issubclass(judge_class, JudgeInterface):
            raise ValueError(f"{judge_class} must be a subclass of JudgeInterface")
        cls._judge_classes[api_type] = judge_class


# Convenience function for backward compatibility
def get_judge(
    api_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> JudgeInterface:
    """
    Get a judge instance with optional configuration
    
    Args:
        api_type: API type ('openai' or 'azure')
        model_name: Model to use for evaluation
        **kwargs: Additional configuration parameters
        
    Returns:
        JudgeInterface instance
    """
    config = None
    if model_name or kwargs:
        config = JudgeConfig(
            model_name=model_name or "gpt-4",
            **kwargs
        )
    
    return JudgeFactory.create_judge(api_type=api_type, config=config)
"""
Additional judge provider implementations (Anthropic, Cohere, Together AI)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger as eval_logger

from .judge import JudgeConfig, JudgeInterface, JudgeRequest, JudgeResponse, OpenAIJudge


class AnthropicJudge(JudgeInterface):
    """Anthropic Claude implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.anthropic_version = "2023-06-01"

        # Try to use Anthropic client if available
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.use_client = True
        except ImportError:
            eval_logger.warning("Anthropic client not available, falling back to requests")
            self.use_client = False

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Anthropic API"""
        if not self.is_available():
            raise ValueError("Anthropic API key not configured")

        config = request.config or self.config
        messages = self.prepare_messages(request)

        # Convert OpenAI format to Anthropic format
        anthropic_messages = self._convert_to_anthropic_format(messages)

        # Extract system prompt if present
        system_prompt = None
        if anthropic_messages and anthropic_messages[0]["role"] == "system":
            system_prompt = anthropic_messages[0]["content"]
            anthropic_messages = anthropic_messages[1:]

        # Prepare payload
        payload = {
            "model": config.model_name,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if config.top_p is not None:
            payload["top_p"] = config.top_p

        # Make API call with retries
        for attempt in range(config.num_retries):
            try:
                if self.use_client:
                    response = self.client.messages.create(**payload)
                    content = response.content[0].text
                    model_used = response.model
                    usage = {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens, "total_tokens": response.usage.input_tokens + response.usage.output_tokens}
                    raw_response = response
                else:
                    response = self._make_request(payload, config.timeout)
                    content = response["content"][0]["text"]
                    model_used = response["model"]
                    usage = response.get("usage")
                    raw_response = response

                return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

            except Exception as e:
                eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise

    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """Make HTTP request to Anthropic API"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _convert_to_anthropic_format(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI message format to Anthropic format"""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # System messages are handled separately in Anthropic
                anthropic_messages.append(msg)
            elif msg["role"] == "user" or msg["role"] == "assistant":
                # Handle content that might include images
                if isinstance(msg["content"], list):
                    anthropic_content = []
                    for item in msg["content"]:
                        if item["type"] == "text":
                            anthropic_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            # Extract base64 data
                            image_data = item["image_url"]["url"]
                            if image_data.startswith("data:image"):
                                # Remove data URL prefix
                                base64_data = image_data.split(",")[1]
                                anthropic_content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_data}})
                    anthropic_messages.append({"role": msg["role"], "content": anthropic_content})
                else:
                    # Simple text content
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        return anthropic_messages


class CohereJudge(JudgeInterface):
    """Cohere implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("COHERE_API_KEY", "")
        self.api_url = "https://api.cohere.ai/v1/chat"

        # Try to use Cohere client if available
        try:
            import cohere

            self.client = cohere.Client(self.api_key)
            self.use_client = True
        except ImportError:
            eval_logger.warning("Cohere client not available, falling back to requests")
            self.use_client = False

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Cohere API"""
        if not self.is_available():
            raise ValueError("Cohere API key not configured")

        config = request.config or self.config
        messages = self.prepare_messages(request)

        # Convert to Cohere format
        message_text = self._convert_to_cohere_format(messages)

        # Prepare payload
        payload = {
            "model": config.model_name,
            "message": message_text,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        if config.top_p is not None:
            payload["p"] = config.top_p

        # Make API call with retries
        for attempt in range(config.num_retries):
            try:
                if self.use_client:
                    response = self.client.chat(**payload)
                    content = response.text
                    model_used = config.model_name
                    usage = (
                        {"prompt_tokens": response.meta.tokens.input_tokens, "completion_tokens": response.meta.tokens.output_tokens, "total_tokens": response.meta.tokens.input_tokens + response.meta.tokens.output_tokens}
                        if hasattr(response, "meta")
                        else None
                    )
                    raw_response = response
                else:
                    response = self._make_request(payload, config.timeout)
                    content = response["text"]
                    model_used = config.model_name
                    usage = None
                    raw_response = response

                return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

            except Exception as e:
                eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise

    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """Make HTTP request to Cohere API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _convert_to_cohere_format(self, messages: List[Dict]) -> str:
        """Convert OpenAI message format to Cohere format"""
        # Cohere uses a simple message string, so we concatenate messages
        formatted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, list):
                # Extract text from content list
                text_parts = [item["text"] for item in content if item["type"] == "text"]
                content = " ".join(text_parts)

            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")

        return "\n\n".join(formatted_messages)


class TogetherAIJudge(JudgeInterface):
    """Together AI implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("TOGETHER_API_KEY", "")
        self.api_url = "https://api.together.xyz/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Together AI API"""
        if not self.is_available():
            raise ValueError("Together AI API key not configured")

        config = request.config or self.config
        messages = self.prepare_messages(request)

        # Together AI uses OpenAI-compatible format
        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        if config.top_p is not None:
            payload["top_p"] = config.top_p

        # Make API call with retries
        for attempt in range(config.num_retries):
            try:
                response = self._make_request(payload, config.timeout)
                content = response["choices"][0]["message"]["content"]
                model_used = response["model"]
                usage = response.get("usage")
                raw_response = response

                return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

            except Exception as e:
                eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise

    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """Make HTTP request to Together AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()


# Auto-register providers when module is imported
try:
    from .judge import JudgeFactory

    JudgeFactory.register_judge("anthropic", AnthropicJudge)
    JudgeFactory.register_judge("cohere", CohereJudge)
    JudgeFactory.register_judge("together", TogetherAIJudge)
except ImportError:
    pass

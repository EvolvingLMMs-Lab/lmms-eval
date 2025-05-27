"""
Async implementation of the Unified Judge Interface with support for multiple providers.
This module provides high-throughput async evaluation capabilities.
"""

import abc
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from loguru import logger as eval_logger

from .judge import JudgeConfig, JudgeInterface, JudgeRequest, JudgeResponse, OpenAIJudge

# Configuration for async retry logic
DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_DELAY = 10  # seconds
DEFAULT_CONCURRENT_REQUESTS = 10  # Default concurrency limit


class AsyncJudgeInterface(JudgeInterface):
    """Abstract base class for async judge implementations"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(DEFAULT_CONCURRENT_REQUESTS)

    @abc.abstractmethod
    async def evaluate_async(self, request: JudgeRequest) -> JudgeResponse:
        """
        Asynchronously evaluate the given request and return a response

        Args:
            request: JudgeRequest containing the evaluation context

        Returns:
            JudgeResponse with the evaluation result
        """
        pass

    async def evaluate_batch(self, requests: List[JudgeRequest]) -> List[JudgeResponse]:
        """
        Evaluate multiple requests concurrently

        Args:
            requests: List of JudgeRequests to evaluate

        Returns:
            List of JudgeResponses in the same order as requests
        """
        tasks = [self.evaluate_async(request) for request in requests]
        return await asyncio.gather(*tasks)

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Synchronous wrapper for async evaluation"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.evaluate_async(request))


class AsyncOpenAIJudge(AsyncJudgeInterface):
    """Async OpenAI API implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        # Try to use async OpenAI client if available
        self.use_async_client = False
        try:
            from openai import AsyncOpenAI

            self.async_client = AsyncOpenAI(api_key=self.api_key)
            self.use_async_client = True
        except ImportError:
            eval_logger.warning("AsyncOpenAI client not available, using aiohttp")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def evaluate_async(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using OpenAI API asynchronously"""
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
        async with self.semaphore:
            for attempt in range(config.num_retries):
                try:
                    if self.use_async_client:
                        response = await self.async_client.chat.completions.create(**payload)
                        content = response.choices[0].message.content
                        model_used = response.model
                        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                        raw_response = response
                    else:
                        response = await self._make_async_request(payload, config.timeout)
                        content = response["choices"][0]["message"]["content"]
                        model_used = response["model"]
                        usage = response.get("usage")
                        raw_response = response

                    return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        await asyncio.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise

    async def _make_async_request(self, payload: Dict, timeout: int) -> Dict:
        """Make async HTTP request to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to messages - reuse from base implementation"""
        return OpenAIJudge._add_images_to_messages(self, messages, images)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 - reuse from base implementation"""
        return OpenAIJudge._encode_image(self, image_path)


class AnthropicJudge(AsyncJudgeInterface):
    """Anthropic Claude implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.anthropic_version = "2023-06-01"

        # Try to use Anthropic client if available
        self.use_client = False
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.use_client = True
        except ImportError:
            eval_logger.warning("Anthropic client not available, using aiohttp")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def evaluate_async(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Anthropic API asynchronously"""
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
        async with self.semaphore:
            for attempt in range(config.num_retries):
                try:
                    if self.use_client:
                        response = await self.async_client.messages.create(**payload)
                        content = response.content[0].text
                        model_used = response.model
                        usage = {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens, "total_tokens": response.usage.input_tokens + response.usage.output_tokens}
                        raw_response = response
                    else:
                        response = await self._make_async_request(payload, config.timeout)
                        content = response["content"][0]["text"]
                        model_used = response["model"]
                        usage = response.get("usage")
                        raw_response = response

                    return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        await asyncio.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise

    async def _make_async_request(self, payload: Dict, timeout: int) -> Dict:
        """Make async HTTP request to Anthropic API"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.json()

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


class CohereJudge(AsyncJudgeInterface):
    """Cohere implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("COHERE_API_KEY", "")
        self.api_url = "https://api.cohere.ai/v1/chat"

        # Try to use Cohere client if available
        self.use_client = False
        try:
            import cohere

            self.client = cohere.Client(self.api_key)
            self.async_client = cohere.AsyncClient(self.api_key)
            self.use_client = True
        except ImportError:
            eval_logger.warning("Cohere client not available, using aiohttp")

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def evaluate_async(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Cohere API asynchronously"""
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
        async with self.semaphore:
            for attempt in range(config.num_retries):
                try:
                    if self.use_client:
                        response = await self.async_client.chat(**payload)
                        content = response.text
                        model_used = config.model_name
                        usage = {"prompt_tokens": response.meta.tokens.input_tokens, "completion_tokens": response.meta.tokens.output_tokens, "total_tokens": response.meta.tokens.input_tokens + response.meta.tokens.output_tokens}
                        raw_response = response
                    else:
                        response = await self._make_async_request(payload, config.timeout)
                        content = response["text"]
                        model_used = config.model_name
                        usage = None
                        raw_response = response

                    return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        await asyncio.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise

    async def _make_async_request(self, payload: Dict, timeout: int) -> Dict:
        """Make async HTTP request to Cohere API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.json()

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


class TogetherAIJudge(AsyncJudgeInterface):
    """Together AI implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("TOGETHER_API_KEY", "")
        self.api_url = "https://api.together.xyz/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def evaluate_async(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Together AI API asynchronously"""
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
        async with self.semaphore:
            for attempt in range(config.num_retries):
                try:
                    response = await self._make_async_request(payload, config.timeout)
                    content = response["choices"][0]["message"]["content"]
                    model_used = response["model"]
                    usage = response.get("usage")
                    raw_response = response

                    return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        await asyncio.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise

    async def _make_async_request(self, payload: Dict, timeout: int) -> Dict:
        """Make async HTTP request to Together AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.json()


class AsyncAzureOpenAIJudge(AsyncJudgeInterface):
    """Async Azure OpenAI implementation of the Judge interface"""

    def __init__(self, config: Optional[JudgeConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("AZURE_API_KEY", "")
        self.api_endpoint = os.getenv("AZURE_ENDPOINT", "")
        self.api_version = os.getenv("API_VERSION", "2024-02-15-preview")

        # Try to use async Azure OpenAI client if available
        self.use_async_client = False
        try:
            from openai import AsyncAzureOpenAI

            self.async_client = AsyncAzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_endpoint, api_version=self.api_version)
            self.use_async_client = True
        except ImportError:
            eval_logger.warning("AsyncAzureOpenAI client not available, using aiohttp")

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_endpoint)

    async def evaluate_async(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate using Azure OpenAI API asynchronously"""
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
        async with self.semaphore:
            for attempt in range(config.num_retries):
                try:
                    if self.use_async_client:
                        response = await self.async_client.chat.completions.create(**payload)
                        content = response.choices[0].message.content
                        model_used = response.model
                        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                        raw_response = response
                    else:
                        response = await self._make_async_request(payload, config.timeout)
                        content = response["choices"][0]["message"]["content"]
                        model_used = response["model"]
                        usage = response.get("usage")
                        raw_response = response

                    return JudgeResponse(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

                except Exception as e:
                    eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                    if attempt < config.num_retries - 1:
                        await asyncio.sleep(config.retry_delay)
                    else:
                        eval_logger.error(f"All {config.num_retries} attempts failed")
                        raise

    async def _make_async_request(self, payload: Dict, timeout: int) -> Dict:
        """Make async HTTP request to Azure OpenAI API"""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Construct the full URL
        deployment_name = payload["model"]
        url = f"{self.api_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                return await response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to messages - reuse from base implementation"""
        return OpenAIJudge._add_images_to_messages(self, messages, images)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 - reuse from base implementation"""
        return OpenAIJudge._encode_image(self, image_path)


class AsyncJudgeFactory:
    """Factory for creating async judge instances based on configuration"""

    _judge_classes = {
        "openai": AsyncOpenAIJudge,
        "azure": AsyncAzureOpenAIJudge,
        "anthropic": AnthropicJudge,
        "cohere": CohereJudge,
        "together": TogetherAIJudge,
    }

    @classmethod
    def create_judge(cls, api_type: Optional[str] = None, config: Optional[JudgeConfig] = None) -> AsyncJudgeInterface:
        """
        Create an async judge instance based on API type

        Args:
            api_type: Type of API to use
            config: Configuration for the judge

        Returns:
            AsyncJudgeInterface instance
        """
        if api_type is None:
            api_type = os.getenv("API_TYPE", "openai").lower()

        if api_type not in cls._judge_classes:
            raise ValueError(f"Unknown API type: {api_type}. Supported types: {list(cls._judge_classes.keys())}")

        judge_class = cls._judge_classes[api_type]
        return judge_class(config=config)

    @classmethod
    def register_judge(cls, api_type: str, judge_class: type):
        """Register a new async judge implementation"""
        if not issubclass(judge_class, AsyncJudgeInterface):
            raise ValueError(f"{judge_class} must be a subclass of AsyncJudgeInterface")
        cls._judge_classes[api_type] = judge_class


# High-level async evaluation utilities
async def evaluate_batch_async(requests: List[JudgeRequest], api_type: Optional[str] = None, config: Optional[JudgeConfig] = None, max_concurrent: int = DEFAULT_CONCURRENT_REQUESTS) -> List[JudgeResponse]:
    """
    Evaluate multiple requests concurrently with rate limiting

    Args:
        requests: List of judge requests to evaluate
        api_type: API type to use
        config: Judge configuration
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of judge responses
    """
    judge = AsyncJudgeFactory.create_judge(api_type=api_type, config=config)
    judge.semaphore = asyncio.Semaphore(max_concurrent)
    return await judge.evaluate_batch(requests)


async def evaluate_with_fallback(request: JudgeRequest, primary_api: str = "openai", fallback_apis: List[str] = ["anthropic", "together"], config: Optional[JudgeConfig] = None) -> JudgeResponse:
    """
    Evaluate with automatic fallback to other providers

    Args:
        request: Judge request to evaluate
        primary_api: Primary API to try first
        fallback_apis: List of fallback APIs to try
        config: Judge configuration

    Returns:
        Judge response from the first successful API
    """
    apis_to_try = [primary_api] + fallback_apis

    for api_type in apis_to_try:
        try:
            judge = AsyncJudgeFactory.create_judge(api_type=api_type, config=config)
            if judge.is_available():
                return await judge.evaluate_async(request)
        except Exception as e:
            eval_logger.warning(f"Failed to evaluate with {api_type}: {str(e)}")
            continue

    raise ValueError("All APIs failed or unavailable")

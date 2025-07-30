import asyncio
import os
from typing import Dict, List, Optional, Union

import aiohttp
from loguru import logger as eval_logger

from ..base import AsyncServerInterface
from ..protocol import Request, Response, ServerConfig
from .openai import OpenAIProvider


class AsyncAzureOpenAIProvider(AsyncServerInterface):
    """Async Azure OpenAI API implementation of the Judge interface"""

    def __init__(self, config: Optional[ServerConfig] = None):
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

    async def evaluate_async(self, request: Request) -> Response:
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

                    return Response(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

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
        return OpenAIProvider._add_images_to_messages(self, messages, images)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 - reuse from base implementation"""
        return OpenAIProvider._encode_image(self, image_path)

import base64
import os
import time
from io import BytesIO
from typing import Dict, List, Optional, Union

import requests
from loguru import logger as eval_logger
from PIL import Image

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class OpenAIProvider(ServerInterface):
    """OpenAI API implementation of the Judge interface"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        # Initialize OpenAI client
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            self.use_client = True
        except ImportError:
            eval_logger.warning("OpenAI client not available, falling back to requests")
            self.use_client = False

    def is_available(self) -> bool:
        return bool(self.api_key)

    def evaluate(self, request: Request) -> Response:
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
                    usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                    raw_response = response
                else:
                    response = self._make_request(payload, config.timeout)
                    content = response["choices"][0]["message"]["content"]
                    model_used = response["model"]
                    usage = response.get("usage")
                    raw_response = response

                return Response(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

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

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """Add images to the last user message"""
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
                        messages[i]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                    elif isinstance(image, bytes):
                        # Already base64 encoded
                        messages[i]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image.decode()}"}})
                break

        return messages

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        img = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

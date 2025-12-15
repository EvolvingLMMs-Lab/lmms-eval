"""
Unified OpenAI-compatible backend for VIEScore evaluation.

This module provides a single backend that works with any OpenAI-compatible API,
including OpenAI GPT-4o/GPT-4V, vLLM servers, and other compatible endpoints.

Usage:
    Set environment variables:
        export VIESCORE_API_BASE="https://api.openai.com/v1"  # or vLLM server URL
        export VIESCORE_API_KEY="your-api-key"
        export VIESCORE_MODEL_NAME="gpt-4o"  # or "Qwen/Qwen2.5-VL-72B-Instruct"
"""

import base64
import os
import random
from io import BytesIO
from typing import List, Optional, Union

import numpy as np


def encode_image_to_base64(image) -> str:
    """Convert PIL Image or file path to base64 string."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    else:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class OpenAIBackend:
    """
    Unified OpenAI-compatible backend for VIEScore.

    Works with:
        - OpenAI API (GPT-4o, GPT-4V, etc.)
        - vLLM servers with OpenAI-compatible API
        - Any other OpenAI-compatible endpoint

    Environment variables:
        - VIESCORE_API_BASE: API base URL (default: "https://api.openai.com/v1")
        - VIESCORE_API_KEY: API key (required)
        - VIESCORE_MODEL_NAME: Model name (default: "gpt-4o")
        - VIESCORE_TIMEOUT: Request timeout in seconds (default: 120)
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize OpenAI backend.

        Args:
            api_base: API base URL. Defaults to VIESCORE_API_BASE env var.
            api_key: API key. Defaults to VIESCORE_API_KEY env var.
            model_name: Model name. Defaults to VIESCORE_MODEL_NAME env var or "gpt-4o".
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        self.api_base = api_base or os.getenv("VIESCORE_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("VIESCORE_API_KEY")
        self.model_name = model_name or os.getenv("VIESCORE_MODEL_NAME", "gpt-4o")
        self.timeout = int(os.getenv("VIESCORE_TIMEOUT", str(timeout)))
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("API key is required. Set VIESCORE_API_KEY environment variable or pass api_key argument.")

        print(f"OpenAIBackend initialized: api_base={self.api_base}, model={self.model_name}")

    def prepare_prompt(
        self,
        image_links: Union[List, object] = [],
        text_prompt: str = "",
    ) -> List[dict]:
        """
        Prepare prompt in OpenAI chat format.

        Args:
            image_links: Single image or list of PIL Images / image paths
            text_prompt: Text prompt

        Returns:
            List of messages in OpenAI chat format
        """
        if not isinstance(image_links, list):
            image_links = [image_links]

        content = []

        for img in image_links:
            img_base64 = encode_image_to_base64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

        content.append({"type": "text", "text": text_prompt})

        return [{"role": "user", "content": content}]

    def get_parsed_output(self, messages: List[dict]) -> str:
        """
        Get model output for the given messages.

        Args:
            messages: Messages in OpenAI chat format

        Returns:
            Model response text
        """
        set_seed(42)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return response.choices[0].message.content if response.choices else ""

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower():
                return "rate_limit_exceeded"
            print(f"Error calling OpenAI API: {e}")
            raise


if __name__ == "__main__":
    # Test the OpenAI backend
    import sys

    api_key = os.getenv("VIESCORE_API_KEY")
    if not api_key:
        print("Please set VIESCORE_API_KEY environment variable")
        sys.exit(1)

    print("Testing OpenAI backend...")

    model = OpenAIBackend()

    # Test with a simple text prompt (no image)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello, what model are you?"}]}]

    response = model.get_parsed_output(messages)
    print(f"Response: {response}")

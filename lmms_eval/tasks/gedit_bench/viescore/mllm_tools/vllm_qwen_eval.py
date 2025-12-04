"""
vLLM-based Qwen VL model evaluation backend.

This module provides evaluation using Qwen2.5-VL or Qwen3-VL models served via vLLM.
vLLM provides an OpenAI-compatible API endpoint.

Usage:
    1. Start vLLM server on remote machine:
       python -m vllm.entrypoints.openai.api_server \
           --model Qwen/Qwen2.5-VL-72B-Instruct \
           --port 8000 \
           --tensor-parallel-size 4

    2. Set environment variables:
       export VLLM_API_BASE="http://remote-server:8000/v1"
       export VLLM_MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"  # optional

    3. Use backbone="vllm_qwen" in VIEScore

"""

import base64
import os
import random
from io import BytesIO
from typing import List, Optional

import numpy as np


def encode_image_to_base64(image) -> str:
    """Convert PIL Image to base64 string"""
    if isinstance(image, str):
        # It's a file path
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    else:
        # It's a PIL Image
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class VLLMQwen:
    """
    vLLM-based Qwen VL model for image evaluation.

    Connects to a vLLM server running Qwen2.5-VL or Qwen3-VL and uses
    the OpenAI-compatible API for inference.

    Environment variables:
        - VLLM_API_BASE: Base URL of vLLM server (e.g., "http://localhost:8000/v1")
        - VLLM_API_KEY: API key if required (default: "EMPTY")
        - VLLM_MODEL_NAME: Model name to use (default: auto-detect from server)
        - VLLM_TIMEOUT: Request timeout in seconds (default: 120)
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        """
        Initialize vLLM Qwen client.

        Args:
            api_base: vLLM server base URL. Defaults to VLLM_API_BASE env var.
            api_key: API key if required. Defaults to VLLM_API_KEY env var or "EMPTY".
            model_name: Model name. Defaults to VLLM_MODEL_NAME env var.
            timeout: Request timeout in seconds.
        """
        self.api_base = api_base or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.model_name = model_name or os.getenv("VLLM_MODEL_NAME")
        self.timeout = int(os.getenv("VLLM_TIMEOUT", str(timeout)))

        # Auto-detect model name if not provided
        if not self.model_name:
            self.model_name = self._get_model_name()

        print(f"VLLMQwen initialized: api_base={self.api_base}, model={self.model_name}")

    def _get_model_name(self) -> str:
        """Get model name from vLLM server"""
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
            )
            models = client.models.list()
            if models.data:
                return models.data[0].id
        except Exception as e:
            print(f"Warning: Could not auto-detect model name: {e}")
        return "default"

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        """
        Prepare prompt for Qwen VL model.

        Args:
            image_links: List of PIL Images or image paths
            text_prompt: Text prompt

        Returns:
            List of messages in OpenAI chat format
        """
        if not isinstance(image_links, list):
            image_links = [image_links]

        # Build content list with images and text
        content = []

        for img in image_links:
            img_base64 = encode_image_to_base64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]

        return messages

    def get_parsed_output(self, messages) -> str:
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
                max_tokens=512,
                temperature=0.1,
                top_p=None,
            )

            return response.choices[0].message.content if response.choices else ""

        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            raise


class VLLMQwen25VL(VLLMQwen):
    """vLLM-based Qwen2.5-VL model"""

    def __init__(self, **kwargs) -> None:
        # Set default model name for Qwen2.5-VL if not provided
        if "model_name" not in kwargs and not os.getenv("VLLM_MODEL_NAME"):
            kwargs["model_name"] = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-VL-72B-Instruct")
        super().__init__(**kwargs)


class VLLMQwen3VL(VLLMQwen):
    """vLLM-based Qwen3-VL model"""

    def __init__(self, **kwargs) -> None:
        # Set default model name for Qwen3-VL if not provided
        if "model_name" not in kwargs and not os.getenv("VLLM_MODEL_NAME"):
            kwargs["model_name"] = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-VL-72B-Instruct")
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Test the vLLM Qwen client
    import sys

    # Check if API base is set
    api_base = os.getenv("VLLM_API_BASE")
    if not api_base:
        print("Please set VLLM_API_BASE environment variable")
        print("Example: export VLLM_API_BASE='http://localhost:8000/v1'")
        sys.exit(1)

    print(f"Testing vLLM Qwen client with API base: {api_base}")

    model = VLLMQwen25VL()

    # Test with a simple text prompt (no image)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello, what model are you?"}]}]

    response = model.get_parsed_output(messages)
    print(f"Response: {response}")

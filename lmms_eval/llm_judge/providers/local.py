"""Local vLLM / SGLang provider for the llm_judge framework.

Connects to any OpenAI-compatible local server without requiring an API key.

Environment variables:
    LLM_JUDGE_URL   - Server URL (default: http://localhost:8000/v1/chat/completions)
"""

import os
import time
from typing import Dict, Optional

import requests
from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class LocalProvider(ServerInterface):
    """Local vLLM/SGLang OpenAI-compatible server implementation."""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_url = os.getenv("LLM_JUDGE_URL", "http://localhost:8000/v1/chat/completions")

    def is_available(self) -> bool:
        try:
            resp = requests.get(self.api_url.replace("/chat/completions", "/models"), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def evaluate(self, request: Request) -> Response:
        config = request.config or self.config
        messages = self.prepare_messages(request)

        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p

        for attempt in range(config.num_retries):
            try:
                resp = requests.post(self.api_url, json=payload, timeout=config.timeout)
                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage")

                return Response(
                    content=content.strip(),
                    model_used=data.get("model", config.model_name),
                    usage=usage,
                    raw_response=data,
                )

            except Exception as e:
                eval_logger.warning(f"Local server attempt {attempt + 1}/{config.num_retries} failed: {e}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} local server attempts failed")
                    raise

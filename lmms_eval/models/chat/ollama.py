"""Ollama chat backend.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1, so this
backend inherits the full OpenAI chat implementation and overrides only
loglikelihood, which Ollama supports via the /api/generate endpoint with
``logprobs=True``.

Example usage::

    python -m lmms_eval \\
        --model ollama \\
        --model_args model_version=llava \\
        --tasks mme --limit 8
"""

from __future__ import annotations

import math
import time
from typing import Any, List, Optional, Tuple

import requests as http_requests
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible as OpenAICompatibleChatBase

_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_NO_KEY = "ollama"


@register_model("ollama")
class Ollama(OpenAICompatibleChatBase):
    """Ollama local inference backend (OpenAI-compatible /v1 API)."""

    is_simple = False

    def __init__(
        self,
        model_version: str = "llava",
        model: Optional[str] = None,
        host: str = _OLLAMA_DEFAULT_BASE_URL,
        base_url: Optional[str] = None,
        api_key: str = _OLLAMA_NO_KEY,
        num_concurrent: int = 4,
        **kwargs: Any,
    ) -> None:
        resolved_base_url = base_url or host
        # Derive the Ollama native API root (without /v1) for loglikelihood calls.
        self._ollama_api_base = resolved_base_url.rstrip("/").removesuffix("/v1")
        super().__init__(
            model_version=model_version,
            model=model,
            base_url=resolved_base_url,
            api_key=api_key,
            num_concurrent=num_concurrent,
            azure_openai=False,
            **kwargs,
        )

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of the continuation given the context.

        Uses Ollama's native ``POST /api/generate`` with ``logprobs=True`` so
        we get per-token log-probs for the full prompt+continuation, then
        subtract the prompt-only log-prob to isolate the continuation score.
        """
        results: List[Tuple[float, bool]] = []
        url = f"{self._ollama_api_base}/api/generate"

        for instance in tqdm(requests, disable=(self.rank != 0), desc="Loglikelihood"):
            context, continuation = instance.args[0], instance.args[1]
            full_text = context + continuation

            def _get_logprob(prompt: str) -> float:
                payload = {
                    "model": self.model_version,
                    "prompt": prompt,
                    "stream": False,
                    "logprobs": True,
                    "options": {"temperature": 0},
                }
                for attempt in range(self.max_retries):
                    try:
                        resp = http_requests.post(url, json=payload, timeout=self.timeout * 6)
                        resp.raise_for_status()
                        data = resp.json()
                        token_lps = data.get("logprobs") or []
                        return float(sum(token_lps))
                    except Exception as exc:
                        eval_logger.info(f"loglikelihood attempt {attempt + 1}/{self.max_retries} failed: {exc}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_backoff_s)
                return -math.inf

            full_lp = _get_logprob(full_text)
            if math.isinf(full_lp):
                results.append((-math.inf, False))
                continue
            ctx_lp = _get_logprob(context) if context else 0.0
            continuation_lp = full_lp - ctx_lp
            is_greedy = continuation_lp >= 0.0
            results.append((continuation_lp, is_greedy))

        return results

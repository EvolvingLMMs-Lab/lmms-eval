"""Ollama chat backend.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1, so this
backend inherits the OpenAI chat implementation for generation.

Example usage::

    python -m lmms_eval \\
        --model ollama \\
        --model_args model_version=llava \\
        --tasks mme --limit 8
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

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
        """Ollama does not expose prompt-token log-likelihoods.

        Ollama's native ``POST /api/generate`` can return ``logprobs`` for
        generated tokens, but lmms-eval's loglikelihood API needs the
        likelihood of a provided continuation under a fixed context. Returning
        a fabricated score would make multiple-choice likelihood tasks look
        valid while producing misleading metrics.
        """
        raise NotImplementedError("Ollama loglikelihood is not supported; use generate_until tasks instead.")

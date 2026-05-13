"""Chat-template variant of the LiteLLM backend.

Inherits the richer ``generate_until`` from ``chat/openai.py`` (ThreadPoolExecutor +
adaptive concurrency + prefix-aware queue). The client is swapped for a LiteLLM-backed
shim in ``__init__`` so every call routes through ``litellm.completion``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible as OpenAICompatibleChatBase
from lmms_eval.models.simple.litellm import _PLACEHOLDER_API_KEY, _LiteLLMClientShim


@register_model("litellm")
class LiteLLMCompatible(OpenAICompatibleChatBase):
    """LiteLLM-backed chat backend."""

    is_simple = False

    def __init__(
        self,
        model_version: str = "openai/gpt-4o-mini",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("OPENAI_API_BASE")

        super().__init__(
            model_version=model_version,
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key or _PLACEHOLDER_API_KEY,
            azure_openai=False,
            **kwargs,
        )
        self.client = _LiteLLMClientShim(api_key=resolved_api_key, base_url=resolved_base_url)

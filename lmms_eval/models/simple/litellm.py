"""LiteLLM backend — routes requests through ``litellm.completion()`` to 100+ providers
(OpenAI, Anthropic, Bedrock, Vertex, Gemini, Ollama, OpenRouter, Groq, DeepSeek, etc.)
using provider-native API keys.

Reuses the ``OpenAICompatible`` simple-backend implementation end-to-end by swapping
``self.client`` for a thin duck-typed shim that dispatches ``chat.completions.create``
to ``litellm.completion``. All retry/concurrency/media-encoding logic is inherited
unchanged.

See https://docs.litellm.ai/docs/providers for the full provider list and model-name
prefix convention (e.g. ``anthropic/claude-3-5-sonnet-20241022``).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.openai import OpenAICompatible as OpenAICompatibleBase

# Placeholder key passed to ``openai.OpenAI(api_key=...)`` inside the super().__init__()
# call. openai-python raises at construction if both the argument and OPENAI_API_KEY env
# var are unset. We replace ``self.client`` with a LiteLLM-backed shim immediately after,
# so this placeholder is never used on the wire.
_PLACEHOLDER_API_KEY = "sk-litellm-placeholder"


class _LiteLLMChatCompletions:
    """Duck-typed ``openai.OpenAI().chat.completions`` surface backed by ``litellm.completion``."""

    def __init__(self, api_key: Optional[str], base_url: Optional[str]) -> None:
        self._api_key = api_key
        self._base_url = base_url

    def create(self, **kwargs: Any) -> Any:
        import litellm  # lazy import — ``litellm`` is an optional extra

        if self._api_key is not None:
            kwargs.setdefault("api_key", self._api_key)
        if self._base_url is not None:
            kwargs.setdefault("api_base", self._base_url)
        return litellm.completion(**kwargs)


class _LiteLLMChat:
    def __init__(self, completions: _LiteLLMChatCompletions) -> None:
        self.completions = completions


class _LiteLLMClientShim:
    """Minimal OpenAI-client shape used by lmms-eval's ``generate_until`` call path.

    lmms-eval only calls ``self.client.chat.completions.create(**payload)``; this shim
    dispatches that single method to ``litellm.completion``. API key / base URL are
    forwarded per-call; when they are None, LiteLLM resolves credentials from
    provider-specific env vars (``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ``AWS_*``, ...).
    """

    def __init__(self, api_key: Optional[str], base_url: Optional[str]) -> None:
        self.chat = _LiteLLMChat(_LiteLLMChatCompletions(api_key=api_key, base_url=base_url))


@register_model("litellm")
class LiteLLMCompatible(OpenAICompatibleBase):
    """LiteLLM-backed backend that inherits OpenAI-compatible batching/retry logic.

    Users select this backend via ``--model litellm --model_args model=<prefixed_name>``.
    Provider API keys come from the user's environment (``ANTHROPIC_API_KEY``, ...) or
    can be passed explicitly via ``--model_args api_key=...``.
    """

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

        # The parent __init__ builds an openai.OpenAI(...) client and raises if no API
        # key is resolvable. Hand it a placeholder so construction succeeds; we swap
        # self.client immediately afterward with a LiteLLM-backed shim that never uses
        # the OpenAI client.
        super().__init__(
            model_version=model_version,
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key or _PLACEHOLDER_API_KEY,
            azure_openai=False,
            **kwargs,
        )
        self.client = _LiteLLMClientShim(api_key=resolved_api_key, base_url=resolved_base_url)

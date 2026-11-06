"""Resolve OpenAI-compatible judge/model endpoints.

Lab default is OpenRouter. Explicit ``OPENAI_API_URL`` or ``OPENAI_API_BASE``
wins; otherwise we use OpenRouter. Both env names are accepted because models
historically read ``OPENAI_API_BASE`` while judges read ``OPENAI_API_URL``.
"""

from __future__ import annotations

import os

DEFAULT_OPENAI_COMPAT_BASE = "https://openrouter.ai/api/v1"


def resolve_openai_api_base() -> str:
    """Return the API root (no ``/chat/completions`` suffix)."""
    raw = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_API_BASE") or DEFAULT_OPENAI_COMPAT_BASE
    base = raw.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base.rstrip("/") or DEFAULT_OPENAI_COMPAT_BASE


def resolve_openai_chat_completions_url() -> str:
    """Return a full ``.../chat/completions`` URL for raw HTTP callers."""
    return f"{resolve_openai_api_base()}/chat/completions"

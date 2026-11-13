"""Resolve OpenAI-compatible endpoints without coupling model and judge routing."""

from __future__ import annotations

import os

DEFAULT_OPENAI_COMPAT_BASE = "https://api.openai.com/v1"


def normalize_openai_api_base(raw: str) -> str:
    """Return an API root without a trailing chat-completions path."""
    base = raw.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base.rstrip("/") or DEFAULT_OPENAI_COMPAT_BASE


def resolve_openai_api_base() -> str:
    """Resolve judges: API_URL, then API_BASE alias, then official OpenAI."""
    raw = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_API_BASE") or DEFAULT_OPENAI_COMPAT_BASE
    return normalize_openai_api_base(raw)


def resolve_openai_model_base() -> str | None:
    """Resolve models from API_BASE, leaving BASE_URL handling to the SDK.

    OPENAI_API_URL belongs to judges and must not redirect model requests.
    """
    raw = os.getenv("OPENAI_API_BASE")
    return normalize_openai_api_base(raw) if raw else None


def resolve_openai_chat_completions_url() -> str:
    """Return a full judge URL for raw HTTP callers."""
    return f"{resolve_openai_api_base()}/chat/completions"

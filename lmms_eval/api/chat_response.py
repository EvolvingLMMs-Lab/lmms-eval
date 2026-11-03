"""Guards for OpenAI-compatible chat completion payloads.

Empty ``choices``, a missing ``message``, and missing or blank ``content`` are
the same failure. Callers that already catch exceptions keep their existing
fallback. ``str.strip`` is only reached for text that is already non-empty.
"""

from __future__ import annotations

from typing import Any

EMPTY_CHAT_RESPONSE = "LLM returned empty or filtered response"


def _lookup(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _text_from_parts(parts: list | tuple) -> str:
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif isinstance(part.get("content"), str):
            texts.append(part["content"])
    return "".join(texts)


def _is_blank(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, (list, tuple)):
        return len(content) == 0
    return False


def require_chat_message_content(response: Any) -> Any:
    """Return the first choice's message content.

    Raises:
        ValueError: choices are missing, the message is missing, or content is
            missing or empty. Non-empty part lists are returned unchanged.
    """
    choices = _lookup(response, "choices")
    if not choices:
        raise ValueError(EMPTY_CHAT_RESPONSE)
    message = _lookup(choices[0], "message")
    if message is None:
        raise ValueError(EMPTY_CHAT_RESPONSE)
    content = _lookup(message, "content")
    if _is_blank(content):
        raise ValueError(EMPTY_CHAT_RESPONSE)
    return content


def require_chat_message_text(response: Any) -> str:
    """Return the first choice as text.

    Blank strings and content that is None raise before any ``str.strip`` call.
    """
    content = require_chat_message_content(response)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        text = _text_from_parts(content)
    else:
        text = str(content)
    if text.strip() == "":
        raise ValueError(EMPTY_CHAT_RESPONSE)
    return text

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE_CONFIG_KEYS = {
    "access_key",
    "access_key_id",
    "api_key",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "credentials",
    "hf_token",
    "huggingface_hub_token",
    "huggingfacehub_api_token",
    "private_key",
    "proxy_authorization",
    "secret_key",
    "secret",
    "set_cookie",
    "token",
}
_SENSITIVE_KEY_SUFFIXES = (
    "api_key",
    "_token",
    "_secret",
    "password",
    "_authorization",
    "_credential",
    "_credentials",
    "_cookie",
    "_private_key",
    "_access_key",
    "_access_key_id",
    "_secret_key",
)
_ASSIGNMENT_START_RE = re.compile(r"(?i)(^|--|[?&,;\s{(\[])([A-Za-z_][\w.-]*)=")
_HEADER_START_RE = re.compile(r"(?i)\b(Authorization|Proxy[_.-]Authorization|Cookie|Set[_.-]Cookie)\s*:\s*")
_SIMPLE_CREDENTIAL_VALUE_RE = re.compile(r"[A-Za-z0-9._~+/-]+")
_COOKIE_CREDENTIAL_VALUE_RE = re.compile(r"[A-Za-z0-9._~+/-]+=[A-Za-z0-9._~+/-]+")
_AMBIGUOUS_COOKIE_RE = re.compile(r"(?i)(?:(?:^|[?&,;\s])(?:cookie|set[_.-]?cookie)=|\b(?:Cookie|Set[_.-]Cookie)\s*:)[^,\r\n]*;")
_HF_TOKEN_VALUE_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_BEARER_PREFIX_RE = re.compile(r"(?i)\bBearer[ \t]+")
_BEARER_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT_SEQUENCE_RE = re.compile(r"[}\])]*(?:,\s*[A-Za-z_][\w.-]*=(?:\[REDACTED\]|[^,;\s}\])]+))+[}\])]*\s*")
_QUERY_ASSIGNMENT_SEQUENCE_RE = re.compile(r"[}\])]*[?&]\s*[A-Za-z_][\w.-]*=(?:\[REDACTED\]|[^?&,;\s}\])]+)(?:[?&]\s*[A-Za-z_][\w.-]*=(?:\[REDACTED\]|[^?&,;\s}\])]+))*[}\])]*\s*")
_COLON_FIELD_SEQUENCE_RE = re.compile(r"""(?x)(?:,\s*["']?[^"'{}\[\],:=\s]+["']?\s*:\s*(?:"[^"\r\n]*"|'[^'\r\n]*'|[^,{}\[\]\s]+))+[}\])]*\s*""")
_CLOSING_CONTAINER_RE = re.compile(r"[}\])]+\s*")
_OPENAI_KEY_VALUE_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")
_AWS_ACCESS_KEY_VALUE_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@")
_PRIVATE_KEY_BLOCK_RE = re.compile(r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL)
_PRIVATE_KEY_MARKER_RE = re.compile(r"-----BEGIN [^-\r\n]*PRIVATE KEY-----", re.IGNORECASE)
_COLON_KEY_DELIMITERS = frozenset("?&,;:= \t\r\n{[()")
_AUTHORIZATION_KEYS = {"authorization", "proxy_authorization", "bearer"}


def _normalized_sensitive_key(key: Any) -> str:
    return re.sub(r"[.-]", "_", str(key).casefold())


def _is_sensitive_key(key: Any) -> bool:
    normalized = _normalized_sensitive_key(key)
    return normalized in _SENSITIVE_CONFIG_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def _validate_credential_boundary(value: str, end: int) -> None:
    remainder = value[end:]
    if not remainder or remainder.isspace() or _ASSIGNMENT_SEQUENCE_RE.fullmatch(remainder) or _QUERY_ASSIGNMENT_SEQUENCE_RE.fullmatch(remainder) or _CLOSING_CONTAINER_RE.fullmatch(remainder):
        return
    raise ValueError("credential-bearing value has an ambiguous boundary")


def _bearer_value_end(value: str, start: int) -> tuple[int, bool]:
    prefix = _BEARER_PREFIX_RE.match(value, start)
    if prefix is None:
        raise ValueError("credential-bearing authorization value cannot be normalized safely")
    if value.startswith("[REDACTED]", prefix.end()):
        end = prefix.end() + len("[REDACTED]")
        redacted = True
    else:
        match = _BEARER_TOKEN_RE.match(value, prefix.end())
        if match is None:
            raise ValueError("credential-bearing Bearer value cannot be normalized safely")
        end = match.end()
        redacted = False
    _validate_credential_boundary(value, end)
    return end, redacted


def _credential_value_end(value: str, start: int, key: str) -> int:
    if value.startswith("[REDACTED]", start):
        end = start + len("[REDACTED]")
        _validate_credential_boundary(value, end)
        return end
    if key in _AUTHORIZATION_KEYS:
        end, _ = _bearer_value_end(value, start)
        return end
    match = (_COOKIE_CREDENTIAL_VALUE_RE if key in {"cookie", "set_cookie"} else _SIMPLE_CREDENTIAL_VALUE_RE).match(value, start)
    if match is None or value[start] in "{[(\"'":
        raise ValueError("credential-bearing structured value cannot be normalized safely")
    end = match.end()
    if end < len(value) and value[end].isspace():
        raise ValueError("credential-bearing free-form value cannot be normalized safely")
    if end < len(value):
        _validate_credential_boundary(value, end)
    return end


def _redact_standalone_bearers(value: str) -> str:
    parts = []
    cursor = 0
    search_from = 0
    while prefix := _BEARER_PREFIX_RE.search(value, search_from):
        end, redacted = _bearer_value_end(value, prefix.start())
        if redacted:
            parts.append(value[cursor:end])
        else:
            parts.extend((value[cursor : prefix.end()], "[REDACTED]"))
        cursor = end
        search_from = end
    parts.append(value[cursor:])
    return "".join(parts)


def _redact_located_credentials(value: str, locator: re.Pattern, *, key_group: int, sensitive_only: bool) -> str:
    parts = []
    cursor = 0
    search_from = 0
    while match := locator.search(value, search_from):
        key = _normalized_sensitive_key(match.group(key_group))
        if sensitive_only and not _is_sensitive_key(key):
            search_from = match.end()
            continue
        end = _credential_value_end(value, match.end(), key)
        parts.extend((value[cursor : match.end()], "[REDACTED]"))
        cursor = end
        search_from = end
    parts.append(value[cursor:])
    return "".join(parts)


def _validate_colon_remainder(remainder: str) -> None:
    remainder = remainder.lstrip()
    if not remainder or _ASSIGNMENT_SEQUENCE_RE.fullmatch(remainder) or _QUERY_ASSIGNMENT_SEQUENCE_RE.fullmatch(remainder) or _COLON_FIELD_SEQUENCE_RE.fullmatch(remainder) or _CLOSING_CONTAINER_RE.fullmatch(remainder):
        return
    raise ValueError("credential-bearing value has an ambiguous boundary")


def _validate_colon_credentials(value: str) -> None:
    for colon, character in enumerate(value):
        if character != ":":
            continue
        key_end = colon
        while key_end and value[key_end - 1].isspace():
            key_end -= 1
        key_start = key_end
        while key_start and value[key_start - 1] not in _COLON_KEY_DELIMITERS:
            key_start -= 1
        key = value[key_start:key_end].strip("\"'")
        if not key or not _is_sensitive_key(key):
            continue
        start = colon + 1
        while start < len(value) and value[start].isspace():
            start += 1
        if key in _AUTHORIZATION_KEYS and _BEARER_PREFIX_RE.match(value, start):
            _bearer_value_end(value, start)
            continue
        quote = value[start] if start < len(value) and value[start] in "\"'" else ""
        marker_start = start + bool(quote)
        if not value.startswith("[REDACTED]", marker_start):
            raise ValueError("credential-bearing value cannot be normalized safely")
        end = marker_start + len("[REDACTED]")
        if quote:
            while end < len(value) and value[end].isspace():
                end += 1
            if end == len(value) or value[end] != quote:
                raise ValueError("credential-bearing value cannot be normalized safely")
            end += 1
        _validate_colon_remainder(value[end:])


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: "[REDACTED]" if _is_sensitive_key(key) else redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        if _AMBIGUOUS_COOKIE_RE.search(value):
            raise ValueError("ambiguous credential-bearing cookie value cannot be normalized safely")
        value = _PRIVATE_KEY_BLOCK_RE.sub("[REDACTED]", value)
        value = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", value)
        value = _redact_located_credentials(value, _HEADER_START_RE, key_group=1, sensitive_only=False)
        value = _redact_located_credentials(value, _ASSIGNMENT_START_RE, key_group=2, sensitive_only=True)
        value = _redact_standalone_bearers(value)
        value = _HF_TOKEN_VALUE_RE.sub("[REDACTED]", value)
        value = _OPENAI_KEY_VALUE_RE.sub("[REDACTED]", value)
        value = _AWS_ACCESS_KEY_VALUE_RE.sub("[REDACTED]", value)
        if any(_is_sensitive_key(match.group(1)) and not value.startswith("[REDACTED]", match.end()) for match in re.finditer(r"([\w.-]+)=", value)):
            raise ValueError("credential-bearing assignment cannot be normalized safely")
        for match in re.finditer(r"([\w.-]+)=", value):
            if _is_sensitive_key(match.group(1)):
                _validate_credential_boundary(value, match.end() + len("[REDACTED]"))
        _validate_colon_credentials(value)
        if _PRIVATE_KEY_MARKER_RE.search(value):
            raise ValueError("credential-bearing value cannot be normalized safely")
        return value
    return value

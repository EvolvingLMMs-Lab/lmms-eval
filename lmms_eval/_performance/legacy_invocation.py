from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from .json_v1 import canonical_json_bytes, validate_v1_json
from .legacy_redaction import redact_secrets

_LEGACY_DIGEST_DOMAIN = b"lmms-eval/BaselinePerformanceRecordV1/safe-legacy-arguments"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _normalize_legacy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(type(key) is str and key.isascii() for key in value):
            raise TypeError("legacy argument keys must be ASCII strings")
        return {key: _normalize_legacy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_legacy_value(item) for item in value]
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("legacy argument strings must be valid UTF-8") from exc
        return value
    if type(value) is bool:
        return value
    if type(value) is int:
        try:
            validate_v1_json(value, path="$legacy", allow_null=False)
        except ValueError as exc:
            raise TypeError("unsupported legacy argument value: int") from exc
        return value
    raise TypeError(f"unsupported legacy argument value: {type(value).__name__}")


def build_legacy_invocation(arguments: Mapping[str, Any], *, digest_only: bool) -> dict[str, Any]:
    sanitized = _normalize_legacy_value(redact_secrets(dict(arguments)))
    if not digest_only:
        return {"kind": "normalized", "arguments": sanitized}
    payload = canonical_json_bytes(sanitized)
    digest = hashlib.sha256(_LEGACY_DIGEST_DOMAIN + b"\0" + payload).hexdigest()
    return {"kind": "digest", "safe_digest": f"sha256:{digest}"}


def validate_legacy_invocation(value: Any) -> None:
    try:
        if type(value) is not dict:
            raise ValueError
        kind = value["kind"]
        if kind == "normalized":
            if set(value) != {"kind", "arguments"} or type(value["arguments"]) is not dict:
                raise ValueError
            if _normalize_legacy_value(value["arguments"]) != value["arguments"] or redact_secrets(value["arguments"]) != value["arguments"]:
                raise ValueError
        elif kind == "digest":
            if set(value) != {"kind", "safe_digest"} or type(value["safe_digest"]) is not str or _SHA256_RE.fullmatch(value["safe_digest"]) is None:
                raise ValueError
        else:
            raise ValueError
        validate_v1_json(value, path="$legacy", allow_null=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid legacy invocation union") from exc

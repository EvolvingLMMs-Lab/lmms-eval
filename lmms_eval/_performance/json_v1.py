from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import rfc8785

_SAFE_INTEGER_MAX = 2**53 - 1


def validate_v1_json(value: Any, *, path: str, allow_null: bool) -> None:
    if value is None:
        if allow_null:
            return
        raise TypeError(f"{path} cannot be null")
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{path} must contain valid UTF-8") from exc
        return
    if type(value) is bool:
        return
    if type(value) is int:
        if not -_SAFE_INTEGER_MAX <= value <= _SAFE_INTEGER_MAX:
            raise ValueError(f"{path} integer is outside the IEEE-754 safe-integer range")
        return
    if type(value) is float:
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
            raise ValueError(f"{path} float must be finite and not negative zero")
        if value.is_integer() and not -_SAFE_INTEGER_MAX <= value <= _SAFE_INTEGER_MAX:
            raise ValueError(f"{path} integer is outside the IEEE-754 safe-integer range")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            validate_v1_json(key, path=f"{path}.<key>", allow_null=False)
            validate_v1_json(item, path=f"{path}.{key}", allow_null=allow_null)
        return
    if isinstance(value, Mapping):
        raise TypeError(f"{path} JSON objects must be concrete dicts")
    if type(value) is list:
        for index, item in enumerate(value):
            validate_v1_json(item, path=f"{path}[{index}]", allow_null=allow_null)
        return
    raise TypeError(f"{path} has unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    validate_v1_json(value, path="$", allow_null=True)
    return rfc8785.dumps(value)

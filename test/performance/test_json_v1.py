import math
from types import MappingProxyType

import pytest

from lmms_eval._performance.json_v1 import canonical_json_bytes, validate_v1_json


class ListSubclass(list):
    pass


def test_canonical_json_bytes_match_rfc8785_number_and_key_order():
    value = {"tiny": 1e-7, "a": 4.5}
    assert canonical_json_bytes(value) == b'{"a":4.5,"tiny":1e-7}'


def test_canonical_json_bytes_use_utf16_key_order_and_no_newline():
    value = {"דּ": "hebrew", "€": "euro", "\r": "CR", "1": "one", "😀": "grin", "\u0080": "control"}
    expected = '{"\\r":"CR","1":"one","\u0080":"control","€":"euro","😀":"grin","דּ":"hebrew"}'.encode("utf-8")
    payload = canonical_json_bytes(value)
    assert payload == expected
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert not payload.endswith(b"\n")


@pytest.mark.parametrize("value", [-(2**53) + 1, 0, 2**53 - 1])
def test_validate_v1_json_accepts_ieee754_safe_integer_boundaries(value):
    validate_v1_json(value, path="$", allow_null=False)


@pytest.mark.parametrize("value", [-(2**53), 2**53, 2**63, -(2**63) - 1])
def test_validate_v1_json_rejects_integers_outside_ieee754_safe_domain(value):
    with pytest.raises(ValueError, match="safe-integer"):
        validate_v1_json(value, path="$", allow_null=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.0])
def test_validate_v1_json_rejects_non_v1_floats(value):
    with pytest.raises(ValueError, match="finite and not negative zero"):
        validate_v1_json(value, path="$.measurement", allow_null=False)


@pytest.mark.parametrize("value", [float(2**53), -float(2**53)])
def test_validate_v1_json_rejects_unsafe_mathematically_integral_floats(value):
    with pytest.raises(ValueError, match="safe-integer"):
        validate_v1_json(value, path="$.measurement", allow_null=False)


def test_validate_v1_json_distinguishes_null_permission():
    validate_v1_json({"measurement": None}, path="$", allow_null=True)
    with pytest.raises(TypeError, match="cannot be null"):
        validate_v1_json(None, path="$.closed", allow_null=False)


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string-key"},
        {"bad": "\ud800"},
        MappingProxyType({"value": 1}),
        (1, 2),
        ListSubclass([1, None]),
        {"bad": object()},
    ],
)
def test_validate_v1_json_rejects_non_json_or_non_concrete_values(value):
    with pytest.raises((TypeError, ValueError)):
        validate_v1_json(value, path="$", allow_null=True)

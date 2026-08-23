from types import MappingProxyType

import pytest

from lmms_eval._performance.legacy_invocation import (
    build_legacy_invocation,
    validate_legacy_invocation,
)


@pytest.mark.parametrize("digest_only", [False, True])
def test_distinct_secrets_collapse_before_normalization(digest_only):
    first = build_legacy_invocation({"api_key": object(), "model_args": "api_key=raw-one,temperature=0"}, digest_only=digest_only)
    second = build_legacy_invocation({"api_key": "raw-two", "model_args": "api_key=raw-two,temperature=0"}, digest_only=digest_only)

    assert first == second


def test_digest_uses_exact_safe_legacy_domain_and_canonical_payload():
    invocation = build_legacy_invocation(
        {"model_args": "api_key=raw-one,temperature=0", "hf_token": "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        digest_only=True,
    )

    assert invocation == {
        "kind": "digest",
        "safe_digest": "sha256:101f972bb4496e4f8108ce7ff0424dc05182c96659b5fa4c3ab3e437c8e517e2",
    }


@pytest.mark.parametrize("digest_only", [False, True])
def test_non_secret_invocation_tail_remains_identity_bearing(digest_only):
    zero = build_legacy_invocation({"model_args": "api_key=raw-zero,temperature=0"}, digest_only=digest_only)
    same_zero = build_legacy_invocation({"model_args": "api_key=different-zero,temperature=0"}, digest_only=digest_only)
    one = build_legacy_invocation({"model_args": "api_key=raw-one,temperature=1"}, digest_only=digest_only)

    assert zero == same_zero
    assert zero != one


@pytest.mark.parametrize("digest_only", [False, True])
def test_ambiguous_credentials_raise_through_parent_redactor(digest_only):
    with pytest.raises(ValueError, match="credential"):
        build_legacy_invocation({"model_args": "authorization=Basic raw"}, digest_only=digest_only)


@pytest.mark.parametrize("digest_only", [False, True])
@pytest.mark.parametrize(
    "model_args",
    [
        "settings=api_key=raw-secret",
        "settings='api_key=raw-secret'",
        "^api_key=first-secret",
        "^cookie=session=first-secret",
        "^api_key=[REDACTED]second-secret",
        "api_key=first-secret^temperature=0",
        "settings={api_key=first-secret,temperature=0}password=second-secret",
    ],
)
def test_union_arms_reject_ambiguous_nested_credentials_without_leaking(model_args, digest_only):
    with pytest.raises(ValueError, match="credential") as exc_info:
        build_legacy_invocation({"model_args": model_args}, digest_only=digest_only)

    assert all(secret not in str(exc_info.value) for secret in ("raw-secret", "first-secret", "second-secret"))


@pytest.mark.parametrize("digest_only", [False, True])
def test_safe_nested_assignment_preserves_only_non_secret_identity(digest_only):
    def capture(secret, temperature):
        return build_legacy_invocation(
            {"model_args": f"settings={{api_key={secret}}},temperature={temperature}"},
            digest_only=digest_only,
        )

    zero = capture("first-secret", 0)
    same_zero = capture("second-secret", 0)
    one = capture("third-secret", 1)

    assert zero == same_zero
    assert zero != one
    assert all(secret not in str((zero, one)) for secret in ("first-secret", "second-secret", "third-secret"))
    if not digest_only:
        assert zero["arguments"]["model_args"] == "settings={api_key=[REDACTED]},temperature=0"
        assert one["arguments"]["model_args"] == "settings={api_key=[REDACTED]},temperature=1"


def test_normalized_validation_requires_already_redacted_arguments():
    with pytest.raises(ValueError, match="invalid legacy invocation union"):
        validate_legacy_invocation({"kind": "normalized", "arguments": {"api_key": "raw-secret"}})

    assert validate_legacy_invocation({"kind": "normalized", "arguments": {"api_key": "[REDACTED]"}}) is None


@pytest.mark.parametrize(
    "invocation",
    [
        {"kind": "normalized", "arguments": {}, "extra": True},
        {"kind": "normalized", "arguments": {"temperature": 0.5}},
        {"kind": "normalized", "arguments": {"api_key": "raw-secret"}},
        {"kind": "normalized", "arguments": {"model_args": "credential:[REDACTED]raw-secret"}},
        {"kind": "digest", "safe_digest": "sha256:bad"},
        {"kind": "digest", "safe_digest": "sha256:" + "a" * 64, "arguments": {}},
        {"kind": "future", "arguments": {}},
        {"kind": "normalized", "arguments": []},
        {"kind": "digest", "safe_digest": b"sha256:" + b"a" * 64},
        {},
        None,
    ],
)
def test_validation_rejects_invalid_union_shapes_kinds_and_digests(invocation):
    with pytest.raises(ValueError, match="invalid legacy invocation union"):
        validate_legacy_invocation(invocation)


@pytest.mark.parametrize("digest_only", [False, True])
@pytest.mark.parametrize(
    "arguments",
    [
        {"mödél": "dummy"},
        {1: "non-string"},
        {"temperature": 0.5},
        {"count": 2**53},
        {"count": -(2**53)},
        {"value": object()},
        {"value": "\ud800"},
    ],
)
def test_build_rejects_values_outside_legacy_projection(arguments, digest_only):
    with pytest.raises((TypeError, ValueError), match="legacy argument"):
        build_legacy_invocation(arguments, digest_only=digest_only)


@pytest.mark.parametrize("digest_only", [False, True])
def test_nested_values_round_trip_with_safe_integer_endpoints(digest_only):
    arguments = MappingProxyType(
        {
            "nested": {"items": [True, -(2**53) + 1, 2**53 - 1, ("tuple",)]},
            "api_key": "raw-secret",
        }
    )
    invocation = build_legacy_invocation(arguments, digest_only=digest_only)

    if digest_only:
        assert invocation == {"kind": "digest", "safe_digest": "sha256:64e49193f02efc0c7ea30b972102845b3022b713e942d8fd6e578009b6d97ccd"}
    else:
        assert invocation == {
            "kind": "normalized",
            "arguments": {
                "nested": {"items": [True, -(2**53) + 1, 2**53 - 1, ["tuple"]]},
                "api_key": "[REDACTED]",
            },
        }
    assert validate_legacy_invocation(invocation) is None

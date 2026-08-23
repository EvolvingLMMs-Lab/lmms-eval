import pytest

from lmms_eval._performance.legacy_redaction import redact_secrets


@pytest.mark.parametrize(
    ("access_key", "openai_key", "user", "password", "bearer", "cookie", "private_key"),
    [
        ("AKIAABCDEFGHIJKLMNOP", "sk-abcdefghijklmnopqrstuv", "alice", "first", "bearer-one", "cookie-one", "private-one"),
        ("ASIAQRSTUVWXYZABCDEF", "sk-zyxwvutsrqponmlkjihgfe", "bob", "second", "bearer-two", "cookie-two", "private-two"),
    ],
)
def test_redacts_nested_credentials_and_normalizes_sequences(access_key, openai_key, user, password, bearer, cookie, private_key):
    value = {
        "headers": {"Authorization": f"Bearer {bearer}"},
        "aws_access_key_id": access_key,
        "aws_secret_access_key": password * 10,
        "cookie": f"session={cookie}",
        "pem": f"-----BEGIN PRIVATE KEY-----\n{private_key}\n-----END PRIVATE KEY-----",
        "raw_header": f"Authorization: Bearer {bearer}",
        "model_args": (f"endpoint=https://{user}:{password}@example.com,bearer=Bearer {bearer}," f"key={openai_key},aws={access_key},cookie=session={cookie},temperature=0"),
        "sequence": [f"Bearer {bearer}", ("api_key=raw",)],
    }

    assert redact_secrets(value) == {
        "headers": {"Authorization": "[REDACTED]"},
        "aws_access_key_id": "[REDACTED]",
        "aws_secret_access_key": "[REDACTED]",
        "cookie": "[REDACTED]",
        "pem": "[REDACTED]",
        "raw_header": "Authorization: [REDACTED]",
        "model_args": ("endpoint=https://[REDACTED]@example.com,bearer=[REDACTED]," "key=[REDACTED],aws=[REDACTED],cookie=[REDACTED],temperature=0"),
        "sequence": ["Bearer [REDACTED]", ["api_key=[REDACTED]"]],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"model_args": "api_key=raw-one,temperature=0", "hf_token": "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
            {"model_args": "api_key=[REDACTED],temperature=0", "hf_token": "[REDACTED]"},
        ),
        (
            {"model_args": "api_key=raw-two,temperature=0", "hf_token": "hf_ZYXWVUTSRQPONMLKJIHGFEDCBA"},
            {"model_args": "api_key=[REDACTED],temperature=0", "hf_token": "[REDACTED]"},
        ),
        ("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "[REDACTED]"),
        ("settings={api_key=raw-one}", "settings={api_key=[REDACTED]}"),
        ("settings={api_key=raw-two}", "settings={api_key=[REDACTED]}"),
        ("settings=(api_key=raw-one)", "settings=(api_key=[REDACTED])"),
        ("settings=[api_key=raw-one]", "settings=[api_key=[REDACTED]]"),
        ("settings={api_key=first-secret?temperature=0}", "settings={api_key=[REDACTED]?temperature=0}"),
        ("settings={api_key=first-secret,temperature=0}", "settings={api_key=[REDACTED],temperature=0}"),
        ("settings={api_key=raw-secret},temperature=0", "settings={api_key=[REDACTED]},temperature=0"),
        ("settings={api_key=raw-secret}?temperature=0", "settings={api_key=[REDACTED]}?temperature=0"),
        ("--api_key=raw-one", "--api_key=[REDACTED]"),
        ("--api_key=raw-two", "--api_key=[REDACTED]"),
        ("endpoint=https://example.test?api_key=raw-one&temperature=0", "endpoint=https://example.test?api_key=[REDACTED]&temperature=0"),
        ("endpoint=https://example.test?api_key=raw-two&temperature=0", "endpoint=https://example.test?api_key=[REDACTED]&temperature=0"),
        ("Bearer [REDACTED]   ", "Bearer [REDACTED]   "),
        ("credential:[REDACTED]", "credential:[REDACTED]"),
        ("credential:[REDACTED] ", "credential:[REDACTED] "),
        ("credential:[REDACTED],temperature=0", "credential:[REDACTED],temperature=0"),
        ("settings={credential:[REDACTED]}", "settings={credential:[REDACTED]}"),
        ('settings={"credential":"[REDACTED]"}', 'settings={"credential":"[REDACTED]"}'),
        ('settings={"credential":"[REDACTED]","temperature":0}', 'settings={"credential":"[REDACTED]","temperature":0}'),
        ("credential:[REDACTED]?temperature=0&limit=1", "credential:[REDACTED]?temperature=0&limit=1"),
        (
            'settings={"credential":"[REDACTED]","api.key":"[REDACTED]"}',
            'settings={"credential":"[REDACTED]","api.key":"[REDACTED]"}',
        ),
    ],
)
def test_redacts_complete_credential_forms(value, expected):
    sanitized = redact_secrets(value)
    assert sanitized == expected
    assert redact_secrets(sanitized) == expected


@pytest.mark.parametrize("raw", ["raw-one", "raw-two"])
def test_redacts_distinct_sensitive_assignment_values(raw):
    value = f"aws_access_key_id={raw}-aws,access_key_id={raw}-id,access_key={raw}-access,secret_access_key={raw}-secret-access,credentials={raw}-credentials,secret_key={raw}-secret,session_token={raw}-session,temperature=0"
    assert redact_secrets(value) == "aws_access_key_id=[REDACTED],access_key_id=[REDACTED],access_key=[REDACTED],secret_access_key=[REDACTED],credentials=[REDACTED],secret_key=[REDACTED],session_token=[REDACTED],temperature=0"


@pytest.mark.parametrize("raw", ["raw-one", "raw-two"])
def test_redacts_distinct_separator_normalized_assignment_values(raw):
    value = f"api-key={raw}-api,api.key={raw}-dot,access-key-id={raw}-access,access.key.id={raw}-access-dot,secret-key={raw}-secret,temperature=0"
    assert redact_secrets(value) == "api-key=[REDACTED],api.key=[REDACTED],access-key-id=[REDACTED],access.key.id=[REDACTED],secret-key=[REDACTED],temperature=0"


@pytest.mark.parametrize("raw", ["raw-one", "raw-two"])
def test_redacts_distinct_bearer_values_with_complete_assignment_tail(raw):
    assert redact_secrets(f"authorization=Bearer {raw},temperature=0,limit=1") == "authorization=[REDACTED],temperature=0,limit=1"


@pytest.mark.parametrize(
    ("invocation", "zero_expected", "one_expected"),
    [
        ("Authorization: Bearer {secret},temperature={temperature}", "Authorization: [REDACTED],temperature=0", "Authorization: [REDACTED],temperature=1"),
        ("Cookie: session={secret},temperature={temperature}", "Cookie: [REDACTED],temperature=0", "Cookie: [REDACTED],temperature=1"),
        ("authorization=Bearer {secret},temperature={temperature}", "authorization=[REDACTED],temperature=0", "authorization=[REDACTED],temperature=1"),
        ("cookie=session={secret},temperature={temperature}", "cookie=[REDACTED],temperature=0", "cookie=[REDACTED],temperature=1"),
    ],
)
def test_comma_tail_preserves_non_secret_identity(invocation, zero_expected, one_expected):
    zero = redact_secrets(invocation.format(secret="raw-zero", temperature=0))
    same_zero = redact_secrets(invocation.format(secret="different-zero", temperature=0))
    one = redact_secrets(invocation.format(secret="raw-one", temperature=1))
    assert (zero, same_zero, one) == (zero_expected, zero_expected, one_expected)
    assert zero != one


@pytest.mark.parametrize(
    ("tail", "zero_expected", "one_expected"),
    [
        ("?temperature={temperature}", "api_key=[REDACTED]?temperature=0", "api_key=[REDACTED]?temperature=1"),
        ("?temperature={temperature}&limit=1", "api_key=[REDACTED]?temperature=0&limit=1", "api_key=[REDACTED]?temperature=1&limit=1"),
        ("?temperature={temperature}?limit=1", "api_key=[REDACTED]?temperature=0?limit=1", "api_key=[REDACTED]?temperature=1?limit=1"),
    ],
)
def test_query_tail_preserves_non_secret_identity(tail, zero_expected, one_expected):
    zero = redact_secrets("api_key=raw-zero" + tail.format(temperature=0))
    same_zero = redact_secrets("api_key=different-zero" + tail.format(temperature=0))
    one = redact_secrets("api_key=raw-one" + tail.format(temperature=1))
    assert (zero, same_zero, one) == (zero_expected, zero_expected, one_expected)
    assert zero != one


@pytest.mark.parametrize(
    "value",
    [
        "authorization=Basic raw",
        'authorization=Digest username="u",response="raw"',
        'credentials={"user": "raw"}',
        "authorization=Bearer raw%tail,temperature=0",
        "Authorization: Bearer raw-one,raw-two",
        "authorization=[REDACTED]raw,temperature=0",
        "authorization=Bearer raw,temperature=",
        "authorization=Bearer raw,temperature=0,raw-tail",
        "authorization=Bearer raw,temperature=0 extra-tail",
        "Bearer [REDACTED]raw",
        "api_key=raw?tail",
        "api_key=raw?temperature=",
        "api_key=raw?temperature=0 raw-tail",
        "api_key=raw?temperature=0&limit=",
        "api_key=raw?temperature=0?tail",
        "api_key=outer-secret?1_api_key=inner-secret",
        "pem=-----BEGIN PRIVATE KEY-----\nopaque-value",
        'settings={"credential": "opaque-value"}',
        'settings={"api.key":"raw-secret"}',
        'settings={"api-key":"raw-secret"}',
        'settings={"api_key":"raw-secret"}',
        'settings={"1_api_key":"raw-secret"}',
        'settings={"1_api_key" : "raw-secret"}',
        'settings={"credential:raw-secret}',
        "credential:[REDACTED]raw-secret",
        "credential:[REDACTED] raw-secret",
        'settings={"credential":"[REDACTED]raw-secret"}',
        "settings={api_key=raw-one}tail",
        "settings={api_key=first-secret?temperature=0}password=second-secret",
        "settings={api_key=first-secret,temperature=0}password=second-secret",
        "settings=api_key=raw-secret",
        "settings='api_key=raw-secret'",
        "^api_key=first-secret",
        "^cookie=session=first-secret",
        "api_key=first-secret^temperature=0",
        "api_key=first-secret^temperature=1",
        "^api_key=[REDACTED]second-secret",
        "settings=api_key=[REDACTED]second-secret",
        "settings='api_key=[REDACTED]second-secret'",
        "^cookie=[REDACTED]second-secret",
    ],
)
def test_rejects_ambiguous_or_incomplete_credentials(value):
    with pytest.raises(ValueError, match="credential") as exc_info:
        redact_secrets(value)
    assert all(secret not in str(exc_info.value) for secret in ("raw-secret", "first-secret", "second-secret"))


def test_rejects_ambiguous_cookie_assignment():
    with pytest.raises(ValueError, match="ambiguous credential"):
        redact_secrets("cookie=session=raw;temperature=0")

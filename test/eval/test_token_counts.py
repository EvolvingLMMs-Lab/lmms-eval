"""Tests for per-sample token counts infrastructure.

Covers:
- TokenCounts dataclass and to_dict()
- GenerationResult dataclass
- unwrap_generation_output() with all input types
- Instance.token_counts field alignment
- ResponseCache._extract_cacheable() with GenerationResult
- ResponseCache._is_valid_response() with GenerationResult
"""

import pytest

from lmms_eval.api.instance import (
    GenerationResult,
    Instance,
    TokenCounts,
    unwrap_generation_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(request_type="generate_until", prompt="prompt", doc_id=0, idx=0, task="t"):
    return Instance(
        request_type=request_type,
        arguments=(prompt, {}, None, doc_id, task, "test"),
        idx=idx,
        metadata={"task": task, "doc_id": doc_id, "repeats": 1},
    )


# ===========================================================================
# TokenCounts
# ===========================================================================


def test_token_counts_defaults_are_none():
    """TokenCounts fields default to None."""
    tc = TokenCounts()
    assert tc.input_tokens is None
    assert tc.output_tokens is None
    assert tc.reasoning_tokens is None


def test_token_counts_to_dict_omits_none_fields():
    """to_dict() excludes None fields."""
    tc = TokenCounts(output_tokens=42)
    d = tc.to_dict()
    assert d == {"output_tokens": 42}
    assert "input_tokens" not in d
    assert "reasoning_tokens" not in d


def test_token_counts_to_dict_all_fields():
    """to_dict() includes all non-None fields."""
    tc = TokenCounts(input_tokens=100, output_tokens=50, reasoning_tokens=10)
    d = tc.to_dict()
    assert d == {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 10}


def test_token_counts_to_dict_empty_when_all_none():
    """to_dict() returns empty dict when all fields are None."""
    tc = TokenCounts()
    assert tc.to_dict() == {}


# ===========================================================================
# GenerationResult
# ===========================================================================


def test_generation_result_text_only():
    """GenerationResult with text only, no token counts."""
    gr = GenerationResult(text="hello")
    assert gr.text == "hello"
    assert gr.token_counts is None


def test_generation_result_with_token_counts():
    """GenerationResult with text and token counts."""
    tc = TokenCounts(output_tokens=10)
    gr = GenerationResult(text="hello", token_counts=tc)
    assert gr.text == "hello"
    assert gr.token_counts.output_tokens == 10


# ===========================================================================
# unwrap_generation_output
# ===========================================================================


def test_unwrap_generation_output_plain_string():
    """Plain string input returns text and None token counts."""
    text, tc = unwrap_generation_output("hello")
    assert text == "hello"
    assert tc is None


def test_unwrap_generation_output_generation_result_with_counts():
    """GenerationResult with token counts is unwrapped correctly."""
    tc = TokenCounts(input_tokens=100, output_tokens=50, reasoning_tokens=10)
    gr = GenerationResult(text="answer", token_counts=tc)
    text, result_tc = unwrap_generation_output(gr)
    assert text == "answer"
    assert result_tc is not None
    assert result_tc.input_tokens == 100
    assert result_tc.output_tokens == 50
    assert result_tc.reasoning_tokens == 10


def test_unwrap_generation_output_generation_result_without_counts():
    """GenerationResult without token counts returns None for counts."""
    gr = GenerationResult(text="answer")
    text, tc = unwrap_generation_output(gr)
    assert text == "answer"
    assert tc is None


def test_unwrap_generation_output_tuple_with_token_counts_object():
    """Tuple of (str, TokenCounts) is unwrapped correctly."""
    tc = TokenCounts(output_tokens=25)
    text, result_tc = unwrap_generation_output(("response", tc))
    assert text == "response"
    assert result_tc.output_tokens == 25


def test_unwrap_generation_output_tuple_with_dict():
    """Tuple of (str, dict) converts dict to TokenCounts."""
    meta = {"input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 5}
    text, tc = unwrap_generation_output(("response", meta))
    assert text == "response"
    assert tc is not None
    assert tc.input_tokens == 10
    assert tc.output_tokens == 20
    assert tc.reasoning_tokens == 5


def test_unwrap_generation_output_tuple_with_partial_dict():
    """Tuple with partial dict creates TokenCounts with None fields."""
    meta = {"output_tokens": 15}
    text, tc = unwrap_generation_output(("response", meta))
    assert text == "response"
    assert tc.output_tokens == 15
    assert tc.input_tokens is None
    assert tc.reasoning_tokens is None


def test_unwrap_generation_output_list_pair():
    """Lists of length 2 with str first element work like tuples."""
    text, tc = unwrap_generation_output(["hello", {"output_tokens": 5}])
    assert text == "hello"
    assert tc.output_tokens == 5


def test_unwrap_generation_output_non_string_fallback():
    """Non-string, non-GenerationResult, non-tuple inputs use str()."""
    text, tc = unwrap_generation_output(42)
    assert text == "42"
    assert tc is None


def test_unwrap_generation_output_empty_string():
    """Empty string is handled correctly."""
    text, tc = unwrap_generation_output("")
    assert text == ""
    assert tc is None


def test_unwrap_generation_output_none_input():
    """None input is converted to string 'None'."""
    text, tc = unwrap_generation_output(None)
    assert text == "None"
    assert tc is None


# ===========================================================================
# Instance.token_counts field
# ===========================================================================


def test_instance_token_counts_default_empty_list():
    """Instance.token_counts defaults to empty list."""
    inst = _make_instance()
    assert inst.token_counts == []


def test_instance_token_counts_append_token_counts():
    """Can append TokenCounts to Instance.token_counts."""
    inst = _make_instance()
    tc = TokenCounts(output_tokens=30)
    inst.token_counts.append(tc)
    assert len(inst.token_counts) == 1
    assert inst.token_counts[0].output_tokens == 30


def test_instance_token_counts_append_none():
    """Can append None to Instance.token_counts."""
    inst = _make_instance()
    inst.token_counts.append(None)
    assert len(inst.token_counts) == 1
    assert inst.token_counts[0] is None


def test_instance_token_counts_alignment_with_resps():
    """token_counts and resps stay aligned during evaluation."""
    inst = _make_instance()
    # Simulate what the evaluator does
    outputs = [
        GenerationResult(text="a", token_counts=TokenCounts(output_tokens=10)),
        "b",  # plain string
    ]
    for output in outputs:
        text, tc = unwrap_generation_output(output)
        inst.resps.append(text)
        inst.token_counts.append(tc)

    assert len(inst.resps) == 2
    assert len(inst.token_counts) == 2
    assert inst.resps[0] == "a"
    assert inst.token_counts[0].output_tokens == 10
    assert inst.resps[1] == "b"
    assert inst.token_counts[1] is None


# ===========================================================================
# ResponseCache integration with GenerationResult
# ===========================================================================


def test_response_cache_extract_cacheable_reduces_to_text():
    """ResponseCache._extract_cacheable() reduces GenerationResult to text."""
    from lmms_eval.caching.response_cache import ResponseCache

    gr = GenerationResult(text="cached text", token_counts=TokenCounts(output_tokens=42))
    result = ResponseCache._extract_cacheable(gr)
    assert result == "cached text"


def test_response_cache_extract_cacheable_passthrough_string():
    """ResponseCache._extract_cacheable() passes through plain strings."""
    from lmms_eval.caching.response_cache import ResponseCache

    result = ResponseCache._extract_cacheable("plain string")
    assert result == "plain string"


def test_response_cache_extract_cacheable_passthrough_tuple():
    """ResponseCache._extract_cacheable() passes through tuples."""
    from lmms_eval.caching.response_cache import ResponseCache

    tup = (1.23, True)
    result = ResponseCache._extract_cacheable(tup)
    assert result == tup


def test_response_cache_is_valid_response_generation_result_valid():
    """ResponseCache._is_valid_response() accepts non-empty GenerationResult."""
    from lmms_eval.caching.response_cache import ResponseCache

    gr = GenerationResult(text="hello", token_counts=TokenCounts(output_tokens=5))
    assert ResponseCache._is_valid_response(gr, "generate_until") is True


def test_response_cache_is_valid_response_generation_result_empty():
    """ResponseCache._is_valid_response() rejects empty GenerationResult."""
    from lmms_eval.caching.response_cache import ResponseCache

    gr = GenerationResult(text="", token_counts=TokenCounts(output_tokens=0))
    assert ResponseCache._is_valid_response(gr, "generate_until") is False


def test_response_cache_is_valid_response_generation_result_whitespace():
    """ResponseCache._is_valid_response() rejects whitespace-only GenerationResult."""
    from lmms_eval.caching.response_cache import ResponseCache

    gr = GenerationResult(text="   ", token_counts=TokenCounts(output_tokens=1))
    assert ResponseCache._is_valid_response(gr, "generate_until") is False

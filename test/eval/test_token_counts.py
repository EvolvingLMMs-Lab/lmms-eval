"""Tests for per-sample token counts infrastructure.

Covers:
- TokenCounts dataclass and to_dict()
- GenerationResult dataclass
- unwrap_generation_output() with all input types
- Instance.token_counts field alignment
- ResponseCache._extract_cacheable() with GenerationResult
- ResponseCache._is_valid_response() with GenerationResult
"""

import unittest

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


class TestTokenCounts(unittest.TestCase):
    def test_defaults_are_none(self):
        tc = TokenCounts()
        self.assertIsNone(tc.input_tokens)
        self.assertIsNone(tc.output_tokens)
        self.assertIsNone(tc.reasoning_tokens)

    def test_to_dict_omits_none_fields(self):
        tc = TokenCounts(output_tokens=42)
        d = tc.to_dict()
        self.assertEqual(d, {"output_tokens": 42})
        self.assertNotIn("input_tokens", d)
        self.assertNotIn("reasoning_tokens", d)

    def test_to_dict_all_fields(self):
        tc = TokenCounts(input_tokens=100, output_tokens=50, reasoning_tokens=10)
        d = tc.to_dict()
        self.assertEqual(d, {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 10})

    def test_to_dict_empty_when_all_none(self):
        tc = TokenCounts()
        self.assertEqual(tc.to_dict(), {})


# ===========================================================================
# GenerationResult
# ===========================================================================


class TestGenerationResult(unittest.TestCase):
    def test_text_only(self):
        gr = GenerationResult(text="hello")
        self.assertEqual(gr.text, "hello")
        self.assertIsNone(gr.token_counts)

    def test_with_token_counts(self):
        tc = TokenCounts(output_tokens=10)
        gr = GenerationResult(text="hello", token_counts=tc)
        self.assertEqual(gr.text, "hello")
        self.assertEqual(gr.token_counts.output_tokens, 10)


# ===========================================================================
# unwrap_generation_output
# ===========================================================================


class TestUnwrapGenerationOutput(unittest.TestCase):
    def test_plain_string(self):
        text, tc = unwrap_generation_output("hello")
        self.assertEqual(text, "hello")
        self.assertIsNone(tc)

    def test_generation_result_with_counts(self):
        tc = TokenCounts(input_tokens=100, output_tokens=50, reasoning_tokens=10)
        gr = GenerationResult(text="answer", token_counts=tc)
        text, result_tc = unwrap_generation_output(gr)
        self.assertEqual(text, "answer")
        self.assertIsNotNone(result_tc)
        self.assertEqual(result_tc.input_tokens, 100)
        self.assertEqual(result_tc.output_tokens, 50)
        self.assertEqual(result_tc.reasoning_tokens, 10)

    def test_generation_result_without_counts(self):
        gr = GenerationResult(text="answer")
        text, tc = unwrap_generation_output(gr)
        self.assertEqual(text, "answer")
        self.assertIsNone(tc)

    def test_tuple_with_token_counts_object(self):
        tc = TokenCounts(output_tokens=25)
        text, result_tc = unwrap_generation_output(("response", tc))
        self.assertEqual(text, "response")
        self.assertEqual(result_tc.output_tokens, 25)

    def test_tuple_with_dict(self):
        meta = {"input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 5}
        text, tc = unwrap_generation_output(("response", meta))
        self.assertEqual(text, "response")
        self.assertIsNotNone(tc)
        self.assertEqual(tc.input_tokens, 10)
        self.assertEqual(tc.output_tokens, 20)
        self.assertEqual(tc.reasoning_tokens, 5)

    def test_tuple_with_partial_dict(self):
        meta = {"output_tokens": 15}
        text, tc = unwrap_generation_output(("response", meta))
        self.assertEqual(text, "response")
        self.assertEqual(tc.output_tokens, 15)
        self.assertIsNone(tc.input_tokens)
        self.assertIsNone(tc.reasoning_tokens)

    def test_list_pair(self):
        """Lists of length 2 with str first element should also work."""
        text, tc = unwrap_generation_output(["hello", {"output_tokens": 5}])
        self.assertEqual(text, "hello")
        self.assertEqual(tc.output_tokens, 5)

    def test_non_string_fallback(self):
        """Non-string, non-GenerationResult, non-tuple inputs -> str(output), None."""
        text, tc = unwrap_generation_output(42)
        self.assertEqual(text, "42")
        self.assertIsNone(tc)

    def test_empty_string(self):
        text, tc = unwrap_generation_output("")
        self.assertEqual(text, "")
        self.assertIsNone(tc)

    def test_none_input(self):
        text, tc = unwrap_generation_output(None)
        self.assertEqual(text, "None")
        self.assertIsNone(tc)


# ===========================================================================
# Instance.token_counts field
# ===========================================================================


class TestInstanceTokenCounts(unittest.TestCase):
    def test_default_empty_list(self):
        inst = _make_instance()
        self.assertEqual(inst.token_counts, [])

    def test_append_token_counts(self):
        inst = _make_instance()
        tc = TokenCounts(output_tokens=30)
        inst.token_counts.append(tc)
        self.assertEqual(len(inst.token_counts), 1)
        self.assertEqual(inst.token_counts[0].output_tokens, 30)

    def test_append_none(self):
        inst = _make_instance()
        inst.token_counts.append(None)
        self.assertEqual(len(inst.token_counts), 1)
        self.assertIsNone(inst.token_counts[0])

    def test_alignment_with_resps(self):
        """token_counts and resps should stay aligned."""
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

        self.assertEqual(len(inst.resps), 2)
        self.assertEqual(len(inst.token_counts), 2)
        self.assertEqual(inst.resps[0], "a")
        self.assertEqual(inst.token_counts[0].output_tokens, 10)
        self.assertEqual(inst.resps[1], "b")
        self.assertIsNone(inst.token_counts[1])


# ===========================================================================
# ResponseCache integration with GenerationResult
# ===========================================================================


class TestResponseCacheGenerationResult(unittest.TestCase):
    def test_extract_cacheable_reduces_to_text(self):
        from lmms_eval.caching.response_cache import ResponseCache

        gr = GenerationResult(text="cached text", token_counts=TokenCounts(output_tokens=42))
        result = ResponseCache._extract_cacheable(gr)
        self.assertEqual(result, "cached text")

    def test_extract_cacheable_passthrough_string(self):
        from lmms_eval.caching.response_cache import ResponseCache

        result = ResponseCache._extract_cacheable("plain string")
        self.assertEqual(result, "plain string")

    def test_extract_cacheable_passthrough_tuple(self):
        from lmms_eval.caching.response_cache import ResponseCache

        tup = (1.23, True)
        result = ResponseCache._extract_cacheable(tup)
        self.assertEqual(result, tup)

    def test_is_valid_response_generation_result_valid(self):
        from lmms_eval.caching.response_cache import ResponseCache

        gr = GenerationResult(text="hello", token_counts=TokenCounts(output_tokens=5))
        self.assertTrue(ResponseCache._is_valid_response(gr, "generate_until"))

    def test_is_valid_response_generation_result_empty(self):
        from lmms_eval.caching.response_cache import ResponseCache

        gr = GenerationResult(text="", token_counts=TokenCounts(output_tokens=0))
        self.assertFalse(ResponseCache._is_valid_response(gr, "generate_until"))

    def test_is_valid_response_generation_result_whitespace(self):
        from lmms_eval.caching.response_cache import ResponseCache

        gr = GenerationResult(text="   ", token_counts=TokenCounts(output_tokens=1))
        self.assertFalse(ResponseCache._is_valid_response(gr, "generate_until"))


if __name__ == "__main__":
    unittest.main()

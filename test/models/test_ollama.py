"""Unit tests for the Ollama backend."""

from __future__ import annotations

import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


def _ensure_decord_stub() -> None:
    """Register a fake decord module so optional_import resolves without the package."""
    if "decord" not in sys.modules:
        mod = types.ModuleType("decord")
        mod.VideoReader = mock.MagicMock()
        mod.cpu = mock.MagicMock()
        sys.modules["decord"] = mod


def _fake_accelerator() -> mock.MagicMock:
    acc = mock.MagicMock()
    acc.num_processes = 1
    acc.local_process_index = 0
    acc.device = "cpu"
    return acc


def _make_ollama(model_version: str = "llava", **kwargs):
    _ensure_decord_stub()
    from lmms_eval.models.chat.ollama import Ollama

    with mock.patch("lmms_eval.models.simple.openai.OpenAI"), mock.patch("lmms_eval.models.simple.openai.Accelerator", return_value=_fake_accelerator()):
        return Ollama(model_version=model_version, **kwargs)


def _make_post_mock(responses: list[dict]) -> mock.MagicMock:
    """MagicMock for http_requests.post returning each response dict in sequence."""
    side_effects = []
    for resp in responses:
        r = mock.MagicMock()
        r.raise_for_status = mock.MagicMock()
        r.json.return_value = resp
        side_effects.append(r)
    return mock.MagicMock(side_effect=side_effects)


def _logprob_response(logprobs: list[float]) -> dict:
    return {"logprobs": logprobs, "response": ""}


def _make_instance(context: str, continuation: str) -> SimpleNamespace:
    return SimpleNamespace(args=(context, continuation), rank=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOllamaRegistration(unittest.TestCase):
    def test_registered_as_chat_model(self) -> None:
        from lmms_eval.models import MODEL_REGISTRY_V2

        manifest = MODEL_REGISTRY_V2.get_manifest("ollama")
        self.assertEqual(manifest.model_id, "ollama")
        self.assertEqual(manifest.chat_class_path, "lmms_eval.models.chat.ollama.Ollama")
        self.assertIsNone(manifest.simple_class_path)

    def test_is_simple_false(self) -> None:
        _ensure_decord_stub()
        from lmms_eval.models.chat.ollama import Ollama

        self.assertFalse(Ollama.is_simple)


class TestOllamaInit(unittest.TestCase):
    def test_default_base_url(self) -> None:
        m = _make_ollama()
        self.assertEqual(m._ollama_api_base, "http://localhost:11434")

    def test_custom_host_strips_v1(self) -> None:
        m = _make_ollama(host="http://myserver:11434/v1")
        self.assertEqual(m._ollama_api_base, "http://myserver:11434")

    def test_model_version_stored(self) -> None:
        m = _make_ollama(model_version="mistral")
        self.assertEqual(m.model_version, "mistral")

    def test_num_concurrent_default(self) -> None:
        m = _make_ollama()
        self.assertEqual(m.num_concurrent, 4)


class TestOllamaLoglikelihood(unittest.TestCase):
    def test_loglikelihood_sums_continuation_tokens(self) -> None:
        model = _make_ollama()
        # full (ctx+cont) = -2.0, context-only = -1.0  →  continuation = -1.0
        post_mock = _make_post_mock(
            [
                _logprob_response([-1.0, -0.5, -0.5]),
                _logprob_response([-0.8, -0.2]),
            ]
        )
        instance = _make_instance("The sky is", " blue")
        with mock.patch("lmms_eval.models.chat.ollama.http_requests.post", post_mock):
            results = model.loglikelihood([instance])

        lp, is_greedy = results[0]
        self.assertAlmostEqual(lp, -1.0, places=5)
        self.assertFalse(is_greedy)

    def test_loglikelihood_payload_fields(self) -> None:
        """Verify the POST payload contains the required Ollama API fields."""
        model = _make_ollama(model_version="llava")
        post_mock = _make_post_mock(
            [
                _logprob_response([-1.0]),
                _logprob_response([-0.5]),
            ]
        )
        instance = _make_instance("ctx", " cont")
        with mock.patch("lmms_eval.models.chat.ollama.http_requests.post", post_mock):
            model.loglikelihood([instance])

        first_call_kwargs = post_mock.call_args_list[0].kwargs
        payload = first_call_kwargs["json"]
        self.assertEqual(payload["model"], "llava")
        self.assertTrue(payload["logprobs"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertIn("/api/generate", post_mock.call_args_list[0].args[0])

    def test_loglikelihood_correct_prompts_sent(self) -> None:
        """First call gets full text, second gets context only."""
        model = _make_ollama()
        post_mock = _make_post_mock(
            [
                _logprob_response([-1.0]),
                _logprob_response([-0.5]),
            ]
        )
        instance = _make_instance("The sky is", " blue")
        with mock.patch("lmms_eval.models.chat.ollama.http_requests.post", post_mock):
            model.loglikelihood([instance])

        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(post_mock.call_args_list[0].kwargs["json"]["prompt"], "The sky is blue")
        self.assertEqual(post_mock.call_args_list[1].kwargs["json"]["prompt"], "The sky is")

    def test_loglikelihood_empty_context_skips_second_call(self) -> None:
        model = _make_ollama()
        post_mock = _make_post_mock([_logprob_response([-0.5, -0.5])])
        instance = _make_instance("", "hello")
        with mock.patch("lmms_eval.models.chat.ollama.http_requests.post", post_mock):
            results = model.loglikelihood([instance])

        self.assertEqual(post_mock.call_count, 1)
        lp, _ = results[0]
        self.assertAlmostEqual(lp, -1.0, places=5)

    def test_loglikelihood_returns_neg_inf_on_failure(self) -> None:
        model = _make_ollama()
        model.max_retries = 2
        model.retry_backoff_s = 0.0

        instance = _make_instance("ctx", " cont")
        with mock.patch("lmms_eval.models.chat.ollama.http_requests.post", side_effect=Exception("conn refused")):
            results = model.loglikelihood([instance])

        lp, _ = results[0]
        self.assertEqual(lp, -math.inf)

    def test_loglikelihood_positive_score_is_greedy(self) -> None:
        model = _make_ollama()
        post_mock = _make_post_mock(
            [
                _logprob_response([0.1, 0.2]),
                _logprob_response([-0.5]),
            ]
        )
        instance = _make_instance("ctx", " cont")
        with mock.patch("lmms_eval.models.chat.ollama.http_requests.post", post_mock):
            results = model.loglikelihood([instance])

        _, is_greedy = results[0]
        self.assertTrue(is_greedy)


if __name__ == "__main__":
    unittest.main()

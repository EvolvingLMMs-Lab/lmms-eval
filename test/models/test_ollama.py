"""Unit tests for the Ollama backend."""

from __future__ import annotations

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
    def test_loglikelihood_is_explicitly_unsupported(self) -> None:
        model = _make_ollama()
        instance = _make_instance("The sky is", " blue")

        with self.assertRaisesRegex(NotImplementedError, "generate_until"):
            model.loglikelihood([instance])


if __name__ == "__main__":
    unittest.main()

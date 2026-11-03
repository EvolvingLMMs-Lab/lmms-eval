"""Unit tests for the LiteLLM backend (simple + chat variants)."""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


def _install_litellm_stub() -> mock.MagicMock:
    """Register a fake ``litellm`` module so ``import litellm`` resolves in tests."""
    fake = types.ModuleType("litellm")
    fake.completion = mock.MagicMock(name="litellm.completion")
    sys.modules["litellm"] = fake
    return fake.completion


def _fake_chat_completion(content: str = "hi", prompt_tokens: int = 3, completion_tokens: int = 5) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, completion_tokens_details=None)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    return SimpleNamespace(choices=[choice], usage=usage, id="cmpl-test", model="test")


class TestLiteLLMShim(unittest.TestCase):
    def test_shim_forwards_to_litellm_completion(self):
        completion = _install_litellm_stub()
        completion.return_value = _fake_chat_completion("pong")

        from lmms_eval.models.simple.litellm import _LiteLLMClientShim

        shim = _LiteLLMClientShim(api_key="sk-user", base_url="https://proxy.example/v1")
        resp = shim.chat.completions.create(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "ping"}],
        )

        self.assertEqual(resp.choices[0].message.content, "pong")
        completion.assert_called_once()
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "anthropic/claude-3-5-sonnet-20241022")
        self.assertEqual(kwargs["api_key"], "sk-user")
        self.assertEqual(kwargs["api_base"], "https://proxy.example/v1")

    def test_shim_without_explicit_credentials(self):
        """When no api_key/base_url is set, the shim must not inject them; LiteLLM
        then falls back to provider-specific env vars (ANTHROPIC_API_KEY, ...)."""
        completion = _install_litellm_stub()
        completion.return_value = _fake_chat_completion()

        from lmms_eval.models.simple.litellm import _LiteLLMClientShim

        shim = _LiteLLMClientShim(api_key=None, base_url=None)
        shim.chat.completions.create(model="openai/gpt-4o-mini", messages=[])

        kwargs = completion.call_args.kwargs
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("api_base", kwargs)

    def test_litellm_registered_as_simple_and_chat(self):
        """Confirm the model manifest is discoverable under both backends."""
        from lmms_eval.models import MODEL_REGISTRY_V2

        manifest = MODEL_REGISTRY_V2.get_manifest("litellm")
        self.assertEqual(manifest.model_id, "litellm")
        self.assertEqual(
            manifest.simple_class_path,
            "lmms_eval.models.simple.litellm.LiteLLMCompatible",
        )
        self.assertEqual(
            manifest.chat_class_path,
            "lmms_eval.models.chat.litellm.LiteLLMCompatible",
        )
        self.assertIn("litellm_chat", manifest.aliases)
        self.assertIn("litellm_compatible", manifest.aliases)


if __name__ == "__main__":
    unittest.main()

"""Tests for MiniMax LLM judge providers.

Covers:
  - Temperature clamping
  - Think-tag stripping
  - MiniMaxProvider (sync) construction, availability, evaluate
  - AsyncMiniMaxProvider construction, availability, evaluate_async
  - ProviderFactory registration ('minimax', 'async_minimax')
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lmms_eval.llm_judge.factory import ProviderFactory
from lmms_eval.llm_judge.protocol import Request, Response, ServerConfig
from lmms_eval.llm_judge.providers.minimax import (
    MiniMaxProvider,
    _clamp_temperature,
    _strip_think_tags,
)


# ============================================================================
# Temperature clamping
# ============================================================================


class TestClampTemperature:
    def test_within_range(self):
        assert _clamp_temperature(0.5) == 0.5

    def test_at_lower_bound(self):
        assert _clamp_temperature(0.0) == 0.0

    def test_at_upper_bound(self):
        assert _clamp_temperature(1.0) == 1.0

    def test_below_lower_bound(self):
        assert _clamp_temperature(-0.5) == 0.0

    def test_above_upper_bound(self):
        assert _clamp_temperature(1.5) == 1.0

    def test_high_temperature(self):
        assert _clamp_temperature(2.0) == 1.0


# ============================================================================
# Think-tag stripping
# ============================================================================


class TestStripThinkTags:
    def test_no_tags(self):
        assert _strip_think_tags("Hello world") == "Hello world"

    def test_single_tag(self):
        assert _strip_think_tags("<think>reasoning</think>Answer") == "Answer"

    def test_multiline_tag(self):
        text = "<think>\nline1\nline2\n</think>\nResult"
        assert _strip_think_tags(text) == "Result"

    def test_multiple_tags(self):
        text = "<think>a</think>X<think>b</think>Y"
        assert _strip_think_tags(text) == "XY"

    def test_empty_tag(self):
        assert _strip_think_tags("<think></think>OK") == "OK"

    def test_only_tag(self):
        assert _strip_think_tags("<think>only</think>") == ""


# ============================================================================
# MiniMaxProvider – construction & availability
# ============================================================================


class TestMiniMaxProviderInit:
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
    @patch("lmms_eval.llm_judge.providers.minimax.OpenAI", create=True)
    def test_is_available_with_key(self, mock_openai_cls):
        # Patch the import inside __init__
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            provider = MiniMaxProvider.__new__(MiniMaxProvider)
            provider.config = ServerConfig(model_name="MiniMax-M2.7")
            provider.api_key = "test-key"
            provider.api_url = f"{MiniMaxProvider.MINIMAX_BASE_URL}/chat/completions"
            provider.use_client = True
        assert provider.is_available() is True

    @patch.dict(os.environ, {}, clear=False)
    def test_is_not_available_without_key(self):
        provider = MiniMaxProvider.__new__(MiniMaxProvider)
        provider.config = ServerConfig(model_name="MiniMax-M2.7")
        provider.api_key = ""
        assert provider.is_available() is False


# ============================================================================
# MiniMaxProvider – evaluate (mocked)
# ============================================================================


def _make_mock_response(content="test response", model="MiniMax-M2.7"):
    """Build a mock OpenAI-style chat completion response."""
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 20},
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


class TestMiniMaxProviderEvaluate:
    def _build_provider(self):
        provider = MiniMaxProvider.__new__(MiniMaxProvider)
        provider.config = ServerConfig(model_name="MiniMax-M2.7")
        provider.api_key = "test-key"
        provider.api_url = f"{MiniMaxProvider.MINIMAX_BASE_URL}/chat/completions"
        provider.use_client = True
        provider.client = MagicMock()
        return provider

    def test_evaluate_returns_response(self):
        provider = self._build_provider()
        mock_resp = _make_mock_response("The answer is 42.")
        provider.client.chat.completions.create.return_value = mock_resp

        request = Request(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            config=ServerConfig(model_name="MiniMax-M2.7", num_retries=1),
        )
        result = provider.evaluate(request)

        assert isinstance(result, Response)
        assert result.content == "The answer is 42."
        assert result.model_used == "MiniMax-M2.7"

    def test_evaluate_strips_think_tags(self):
        provider = self._build_provider()
        mock_resp = _make_mock_response("<think>reasoning</think>Final answer.")
        provider.client.chat.completions.create.return_value = mock_resp

        request = Request(
            messages=[{"role": "user", "content": "Think hard."}],
            config=ServerConfig(model_name="MiniMax-M2.7", num_retries=1),
        )
        result = provider.evaluate(request)
        assert result.content == "Final answer."

    def test_evaluate_clamps_temperature(self):
        provider = self._build_provider()
        mock_resp = _make_mock_response("ok")
        provider.client.chat.completions.create.return_value = mock_resp

        request = Request(
            messages=[{"role": "user", "content": "hi"}],
            config=ServerConfig(
                model_name="MiniMax-M2.7", temperature=2.0, num_retries=1
            ),
        )
        provider.evaluate(request)

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0

    def test_evaluate_raises_without_key(self):
        provider = self._build_provider()
        provider.api_key = ""

        request = Request(
            messages=[{"role": "user", "content": "hi"}],
            config=ServerConfig(model_name="MiniMax-M2.7", num_retries=1),
        )
        with pytest.raises(ValueError, match="MiniMax API key not configured"):
            provider.evaluate(request)

    def test_evaluate_retries_on_failure(self):
        provider = self._build_provider()
        provider.client.chat.completions.create.side_effect = [
            RuntimeError("timeout"),
            _make_mock_response("recovered"),
        ]

        request = Request(
            messages=[{"role": "user", "content": "retry?"}],
            config=ServerConfig(
                model_name="MiniMax-M2.7", num_retries=2, retry_delay=0
            ),
        )
        result = provider.evaluate(request)
        assert result.content == "recovered"

    def test_evaluate_with_json_response_format(self):
        provider = self._build_provider()
        mock_resp = _make_mock_response('{"score": 5}')
        provider.client.chat.completions.create.return_value = mock_resp

        request = Request(
            messages=[{"role": "user", "content": "score this"}],
            config=ServerConfig(
                model_name="MiniMax-M2.7",
                response_format="json",
                num_retries=1,
            ),
        )
        provider.evaluate(request)

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_evaluate_with_top_p(self):
        provider = self._build_provider()
        mock_resp = _make_mock_response("ok")
        provider.client.chat.completions.create.return_value = mock_resp

        request = Request(
            messages=[{"role": "user", "content": "hi"}],
            config=ServerConfig(
                model_name="MiniMax-M2.7", top_p=0.9, num_retries=1
            ),
        )
        provider.evaluate(request)

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_evaluate_fallback_requests(self):
        provider = self._build_provider()
        provider.use_client = False

        mock_json = {
            "choices": [{"message": {"content": "fallback answer"}}],
            "model": "MiniMax-M2.7",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

        with patch("lmms_eval.llm_judge.providers.minimax.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_json),
                raise_for_status=MagicMock(),
            )

            request = Request(
                messages=[{"role": "user", "content": "hi"}],
                config=ServerConfig(model_name="MiniMax-M2.7", num_retries=1),
            )
            result = provider.evaluate(request)
            assert result.content == "fallback answer"


# ============================================================================
# AsyncMiniMaxProvider
# ============================================================================


class TestAsyncMiniMaxProvider:
    def _build_async_provider(self):
        from lmms_eval.llm_judge.providers.async_minimax import AsyncMiniMaxProvider

        provider = AsyncMiniMaxProvider.__new__(AsyncMiniMaxProvider)
        provider.config = ServerConfig(model_name="MiniMax-M2.7")
        provider.api_key = "test-key"
        provider.api_url = f"{AsyncMiniMaxProvider.MINIMAX_BASE_URL}/chat/completions"
        provider.use_async_client = True
        provider.async_client = MagicMock()
        provider.semaphore = asyncio.Semaphore(10)
        return provider

    def test_is_available(self):
        provider = self._build_async_provider()
        assert provider.is_available() is True

    def test_is_not_available(self):
        provider = self._build_async_provider()
        provider.api_key = ""
        assert provider.is_available() is False

    def test_evaluate_async(self):
        provider = self._build_async_provider()
        mock_resp = _make_mock_response("async answer")
        provider.async_client.chat.completions.create = AsyncMock(
            return_value=mock_resp
        )

        request = Request(
            messages=[{"role": "user", "content": "async test"}],
            config=ServerConfig(model_name="MiniMax-M2.7", num_retries=1),
        )
        result = asyncio.get_event_loop().run_until_complete(
            provider.evaluate_async(request)
        )
        assert isinstance(result, Response)
        assert result.content == "async answer"

    def test_evaluate_async_strips_think_tags(self):
        provider = self._build_async_provider()
        mock_resp = _make_mock_response("<think>steps</think>Done")
        provider.async_client.chat.completions.create = AsyncMock(
            return_value=mock_resp
        )

        request = Request(
            messages=[{"role": "user", "content": "think"}],
            config=ServerConfig(model_name="MiniMax-M2.7", num_retries=1),
        )
        result = asyncio.get_event_loop().run_until_complete(
            provider.evaluate_async(request)
        )
        assert result.content == "Done"


# ============================================================================
# ProviderFactory registration
# ============================================================================


class TestProviderFactoryMiniMax:
    def test_minimax_registered(self):
        assert "minimax" in ProviderFactory._provider_classes

    def test_async_minimax_registered(self):
        assert "async_minimax" in ProviderFactory._provider_classes

    def test_create_minimax_provider(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k"}, clear=False):
            provider = ProviderFactory.create_provider(
                api_type="minimax",
                config=ServerConfig(model_name="MiniMax-M2.7"),
            )
        assert isinstance(provider, MiniMaxProvider)

    def test_create_async_minimax_provider(self):
        from lmms_eval.llm_judge.providers.async_minimax import AsyncMiniMaxProvider

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k"}, clear=False):
            provider = ProviderFactory.create_provider(
                api_type="async_minimax",
                config=ServerConfig(model_name="MiniMax-M2.7"),
            )
        assert isinstance(provider, AsyncMiniMaxProvider)

    def test_env_api_type_minimax(self):
        with patch.dict(
            os.environ,
            {"API_TYPE": "minimax", "MINIMAX_API_KEY": "k"},
            clear=False,
        ):
            provider = ProviderFactory.create_provider(
                config=ServerConfig(model_name="MiniMax-M2.7")
            )
        assert isinstance(provider, MiniMaxProvider)


# ============================================================================
# Integration tests (skipped without MINIMAX_API_KEY)
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxIntegration:
    """Live integration tests against the real MiniMax API."""

    def test_live_evaluate(self):
        config = ServerConfig(
            model_name="MiniMax-M2.7",
            temperature=0.0,
            max_tokens=256,
            num_retries=2,
        )
        provider = MiniMaxProvider(config=config)
        request = Request(
            messages=[{"role": "user", "content": "Reply with exactly: hello"}],
            config=config,
        )
        result = provider.evaluate(request)
        assert isinstance(result, Response)
        assert result.content  # non-empty
        assert result.model_used

    def test_live_json_response(self):
        config = ServerConfig(
            model_name="MiniMax-M2.7",
            temperature=0.0,
            max_tokens=256,
            response_format="json",
            num_retries=2,
        )
        provider = MiniMaxProvider(config=config)
        request = Request(
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object: {"score": 5}',
                }
            ],
            config=config,
        )
        result = provider.evaluate(request)
        assert "score" in result.content

    def test_live_async_evaluate(self):
        from lmms_eval.llm_judge.providers.async_minimax import AsyncMiniMaxProvider

        config = ServerConfig(
            model_name="MiniMax-M2.7",
            temperature=0.0,
            max_tokens=256,
            num_retries=2,
        )
        provider = AsyncMiniMaxProvider(config=config)
        request = Request(
            messages=[{"role": "user", "content": "Reply with exactly: world"}],
            config=config,
        )
        result = asyncio.get_event_loop().run_until_complete(
            provider.evaluate_async(request)
        )
        assert isinstance(result, Response)
        assert result.content

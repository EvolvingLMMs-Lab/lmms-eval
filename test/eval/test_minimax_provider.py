"""Constructor and normalization tests for the MiniMax judge providers."""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from lmms_eval.llm_judge.factory import ProviderFactory
from lmms_eval.llm_judge.protocol import Request, Response, ServerConfig
from lmms_eval.llm_judge.providers.async_minimax import AsyncMiniMaxProvider
from lmms_eval.llm_judge.providers.minimax import (
    MINIMAX_BASE_URL,
    MiniMaxProvider,
    _clamp_temperature,
    _strip_think_tags,
)


def _completion(content: str, model: str = "MiniMax-M3"):
    usage = SimpleNamespace(
        prompt_tokens=3,
        completion_tokens=2,
        model_dump=lambda: {"prompt_tokens": 3, "completion_tokens": 2},
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _request(content: str = "hi", **config_kwargs) -> Request:
    config_kwargs.setdefault("model_name", "MiniMax-M3")
    config_kwargs.setdefault("num_retries", 1)
    config_kwargs.setdefault("retry_delay", 0)
    return Request(messages=[{"role": "user", "content": content}], config=ServerConfig(**config_kwargs))


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-0.5, 0.0), (1.5, 1.0), (2.0, 1.0)],
)
def test_clamp_temperature(temperature, expected):
    assert _clamp_temperature(temperature) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", "Hello world"),
        ("<think>reasoning</think>Answer", "Answer"),
        ("<think>line\nbreak</think>Result", "Result"),
        ("X<think>a</think>Y", "XY"),
        ("<think></think>OK", "OK"),
        ("<think>only</think>", ""),
    ],
)
def test_strip_think_tags(text, expected):
    assert _strip_think_tags(text) == expected


class TestMiniMaxConstructors:
    def test_environment_only_setup_defaults_to_minimax_m3(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_URL", raising=False)

        provider = MiniMaxProvider()

        assert provider.config.model_name == "MiniMax-M3"
        assert provider.is_available() is True
        assert provider.api_key == "minimax-key"
        assert provider.base_url == MINIMAX_BASE_URL
        assert "api.minimax.io" in str(provider.client.base_url)
        assert provider.client.api_key == "minimax-key"

    def test_openai_credentials_are_not_reused(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("OPENAI_API_URL", "https://api.openai.com/v1")

        provider = MiniMaxProvider()

        assert provider.api_key == "minimax-key"
        assert provider.base_url == MINIMAX_BASE_URL
        assert provider.client.api_key == "minimax-key"

    def test_missing_key_is_unavailable_even_if_openai_key_is_set(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        provider = MiniMaxProvider()

        assert provider.is_available() is False
        assert provider.config.model_name == "MiniMax-M3"

    def test_explicit_config_is_preserved(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        config = ServerConfig(model_name="MiniMax-M2.7", temperature=0.2, max_tokens=32)

        provider = MiniMaxProvider(config)

        assert provider.config is config
        assert provider.config.model_name == "MiniMax-M2.7"
        assert provider.config.temperature == 0.2
        assert provider.config.max_tokens == 32

    def test_async_environment_only_setup(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")

        provider = AsyncMiniMaxProvider()

        assert provider.config.model_name == "MiniMax-M3"
        assert provider.semaphore._value == provider.config.max_concurrent
        assert provider.is_available() is True
        assert provider.api_url == f"{MINIMAX_BASE_URL}/chat/completions"
        assert "api.minimax.io" in str(provider.async_client.base_url)

    def test_async_explicit_config(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        config = ServerConfig(model_name="MiniMax-M2.7-highspeed", max_concurrent=3)

        provider = AsyncMiniMaxProvider(config)

        assert provider.config is config
        assert provider.config.model_name == "MiniMax-M2.7-highspeed"
        assert provider.semaphore._value == 3


class TestMiniMaxEvaluate:
    def _provider(self, monkeypatch) -> MiniMaxProvider:
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        provider = MiniMaxProvider()
        provider.client = mock.Mock()
        provider.use_client = True
        return provider

    def test_explicit_provider_config_is_sent_when_the_request_has_none(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        config = ServerConfig(model_name="MiniMax-M2.7", temperature=0.3, num_retries=1, retry_delay=0)
        provider = MiniMaxProvider(config)
        provider.client = mock.Mock()
        provider.use_client = True
        provider.client.chat.completions.create.return_value = _completion("ok", model="MiniMax-M2.7")

        result = provider.evaluate(Request(messages=[{"role": "user", "content": "hi"}]))

        payload = provider.client.chat.completions.create.call_args.kwargs
        assert payload["model"] == "MiniMax-M2.7"
        assert payload["temperature"] == 0.3
        assert result.content == "ok"
        assert config.temperature == 0.3

    def test_clamps_temperature_without_mutating_caller_config(self, monkeypatch):
        provider = self._provider(monkeypatch)
        provider.client.chat.completions.create.return_value = _completion("ok")
        request = _request(temperature=2.0)

        provider.evaluate(request)

        payload = provider.client.chat.completions.create.call_args.kwargs
        assert payload["temperature"] == 1.0
        assert payload["model"] == "MiniMax-M3"
        assert request.config.temperature == 2.0

    def test_strips_think_tags(self, monkeypatch):
        provider = self._provider(monkeypatch)
        provider.client.chat.completions.create.return_value = _completion("<think>secret</think>Final answer.")

        result = provider.evaluate(_request("Think hard."))

        assert isinstance(result, Response)
        assert result.content == "Final answer."
        assert result.model_used == "MiniMax-M3"

    def test_forwards_json_format_and_top_p(self, monkeypatch):
        provider = self._provider(monkeypatch)
        provider.client.chat.completions.create.return_value = _completion('{"score": 5}')

        provider.evaluate(_request(response_format="json", top_p=0.9))

        payload = provider.client.chat.completions.create.call_args.kwargs
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["top_p"] == 0.9

    def test_retries_then_returns(self, monkeypatch):
        provider = self._provider(monkeypatch)
        provider.client.chat.completions.create.side_effect = [RuntimeError("timeout"), _completion("recovered")]

        result = provider.evaluate(_request(num_retries=2))

        assert result.content == "recovered"
        assert provider.client.chat.completions.create.call_count == 2

    def test_missing_key_names_minimax(self, monkeypatch):
        provider = self._provider(monkeypatch)
        provider.api_key = ""

        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            provider.evaluate(_request())

    def test_requests_fallback_posts_to_minimax(self, monkeypatch):
        provider = self._provider(monkeypatch)
        provider.use_client = False
        response = mock.Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "<think>x</think>fallback answer"}}],
            "model": "MiniMax-M3",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        response.raise_for_status.return_value = None

        with mock.patch("lmms_eval.llm_judge.providers.openai.requests.post", return_value=response) as post:
            result = provider.evaluate(_request())

        assert post.call_args.args[0] == f"{MINIMAX_BASE_URL}/chat/completions"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer minimax-key"
        assert result.content == "fallback answer"


class TestAsyncMiniMaxEvaluate:
    def test_strips_think_tags_and_clamps_temperature(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        provider = AsyncMiniMaxProvider(ServerConfig(model_name="MiniMax-M2.7", temperature=1.4, num_retries=1, retry_delay=0))
        provider.async_client = mock.Mock()
        provider.use_async_client = True

        async def create(**payload):
            assert payload["temperature"] == 1.0
            assert payload["model"] == "MiniMax-M2.7"
            return _completion("<think>hidden</think>async answer", model="MiniMax-M2.7")

        provider.async_client.chat.completions.create = create

        result = asyncio.run(provider.evaluate_async(Request(messages=[{"role": "user", "content": "hi"}])))

        assert result.content == "async answer"
        assert result.model_used == "MiniMax-M2.7"
        assert provider.config.temperature == 1.4


class TestProviderFactoryMiniMax:
    def test_registered(self):
        assert ProviderFactory._provider_classes["minimax"] is MiniMaxProvider
        assert ProviderFactory._provider_classes["async_minimax"] is AsyncMiniMaxProvider

    def test_environment_only_factory(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        monkeypatch.setenv("API_TYPE", "minimax")

        provider = ProviderFactory.create_provider()

        assert isinstance(provider, MiniMaxProvider)
        assert provider.config.model_name == "MiniMax-M3"
        assert provider.is_available() is True

    def test_explicit_config_factory(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        config = ServerConfig(model_name="MiniMax-M2.7")

        provider = ProviderFactory.create_provider(api_type="minimax", config=config)
        async_provider = ProviderFactory.create_provider(api_type="async_minimax", config=config)

        assert isinstance(provider, MiniMaxProvider)
        assert provider.config is config
        assert isinstance(async_provider, AsyncMiniMaxProvider)
        assert async_provider.config.model_name == "MiniMax-M2.7"

from unittest.mock import patch

from lmms_eval.llm_judge import ProviderFactory, ServerConfig
from lmms_eval.llm_judge.providers import AtlasCloudProvider, OpenAIProvider


@patch("openai.OpenAI")
def test_openai_provider_keeps_existing_environment(mock_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_API_URL", "https://openai.example.com/v1")

    provider = OpenAIProvider(ServerConfig(model_name="gpt-4.1-mini"))

    mock_openai.assert_called_once_with(
        api_key="openai-test-key",
        base_url="https://openai.example.com/v1",
    )
    assert provider.is_available()


@patch("openai.OpenAI")
def test_atlascloud_provider_uses_dedicated_environment(mock_openai, monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    monkeypatch.delenv("ATLASCLOUD_API_URL", raising=False)

    provider = AtlasCloudProvider(ServerConfig(model_name="openai/gpt-4.1-mini"))

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.atlascloud.ai/v1",
    )
    assert provider.is_available()


@patch("openai.OpenAI")
def test_atlascloud_provider_is_registered(mock_openai, monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    monkeypatch.setenv("ATLASCLOUD_API_URL", "https://example.com/v1")

    provider = ProviderFactory.create_provider(
        "atlascloud",
        ServerConfig(model_name="qwen/qwen3.5-flash"),
    )

    assert isinstance(provider, AtlasCloudProvider)
    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.com/v1",
    )

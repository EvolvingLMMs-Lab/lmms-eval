import os
from pathlib import Path
from typing import Optional

from .base import ServerInterface
from .protocol import ServerConfig

# Load .env for LLM judge configuration
try:
    from dotenv import load_dotenv

    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[4] / ".env"]:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
except ImportError:
    pass
from .providers import (
    AsyncAzureOpenAIProvider,
    AsyncOpenAIProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    DummyProvider,
    LocalProvider,
    OpenAIProvider,
)


class ProviderFactory:
    """Factory for creating judge instances based on configuration"""

    _provider_classes = {
        "openai": OpenAIProvider,
        "azure": AzureOpenAIProvider,
        "async_openai": AsyncOpenAIProvider,
        "async_azure": AsyncAzureOpenAIProvider,
        "bedrock": BedrockProvider,
        "local": LocalProvider,
        "dummy": DummyProvider,
    }

    # TODO
    # This should actually be a decorator that registers the class
    @classmethod
    def register_additional_providers(cls):
        """Register additional providers if available"""
        pass

    @classmethod
    def create_provider(cls, api_type: Optional[str] = None, config: Optional[ServerConfig] = None) -> ServerInterface:
        """
        Create a judge instance based on API type

        Args:
            api_type: Type of API to use ('openai' or 'azure').
                     If None, will use API_TYPE environment variable
            config: Configuration for the judge

        Returns:
            ServerInterface instance
        """
        if api_type is None:
            api_type = os.getenv("API_TYPE", "openai").lower()

        if api_type not in cls._provider_classes:
            raise ValueError(f"Unknown API type: {api_type}. Supported types: {list(cls._provider_classes.keys())}")

        judge_class = cls._provider_classes[api_type]
        return judge_class(config=config)

    @classmethod
    def register_provider(cls, api_type: str, judge_class: type):
        """Register a new judge implementation"""
        if not issubclass(judge_class, ServerInterface):
            raise ValueError(f"{judge_class} must be a subclass of ServerInterface")
        cls._provider_classes[api_type] = judge_class

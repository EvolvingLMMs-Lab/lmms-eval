import os
from typing import Optional

from .base import ServerInterface
from .protocol import ServerConfig
from .providers import AsyncOpenAIProvider, AzureOpenAIProvider, OpenAIProvider


class ProviderFactory:
    """Factory for creating judge instances based on configuration"""

    _provider_classes = {"openai": OpenAIProvider, "azure": AzureOpenAIProvider, "async_openai": AsyncOpenAIProvider}

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

        if api_type not in cls._judge_classes:
            raise ValueError(f"Unknown API type: {api_type}. Supported types: {list(cls._judge_classes.keys())}")

        judge_class = cls._judge_classes[api_type]
        return judge_class(config=config)

    @classmethod
    def register_provider(cls, api_type: str, judge_class: type):
        """Register a new judge implementation"""
        if not issubclass(judge_class, ServerInterface):
            raise ValueError(f"{judge_class} must be a subclass of ServerInterface")
        cls._provider_classes[api_type] = judge_class


# Convenience function for backward compatibility
def get_provider(api_type: Optional[str] = None, model_name: Optional[str] = None, **kwargs) -> ServerInterface:
    """
    Get a provider instance with optional configuration

    Args:
        api_type: API type ('openai' or 'azure')
        model_name: Model to use for evaluation
        **kwargs: Additional configuration parameters

    Returns:
        ServerInterface instance
    """
    config = None
    if model_name or kwargs:
        config = ServerConfig(model_name=model_name or "gpt-4", **kwargs)

    return ProviderFactory.create_provider(api_type=api_type, config=config)

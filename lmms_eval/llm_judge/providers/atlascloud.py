from .openai import OpenAIProvider


class AtlasCloudProvider(OpenAIProvider):
    """Atlas Cloud implementation of the OpenAI-compatible judge interface."""

    api_key_env = "ATLASCLOUD_API_KEY"
    base_url_env = "ATLASCLOUD_API_URL"
    default_base_url = "https://api.atlascloud.ai/v1"
    provider_name = "Atlas Cloud"

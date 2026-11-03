"""MiniMax judge provider.

MiniMax exposes an OpenAI-compatible chat completions endpoint, so this
provider reuses :class:`OpenAIProvider` and only keeps the MiniMax-specific
defaults: credentials, the MiniMax-M3 model, temperature clamped to
``[0.0, 1.0]``, and ``<think>`` tag stripping for reasoning models.
"""

import os
import re
from dataclasses import replace
from typing import Optional

from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig
from .openai import OpenAIProvider

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M3"
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _clamp_temperature(temperature: float) -> float:
    """Clamp temperature to MiniMax's accepted range [0.0, 1.0]."""
    return max(0.0, min(1.0, temperature))


def _strip_think_tags(text: str) -> str:
    """Strip ``<think>...</think>`` tags that MiniMax reasoning models may emit."""
    return _THINK_TAG_RE.sub("", text).strip()


def _default_config(config: Optional[ServerConfig]) -> ServerConfig:
    """Use MiniMax-M3 when the caller does not pass a config.

    ``ServerInterface`` otherwise fills in ``model_name="gpt-4"``.
    """
    if config is None:
        return ServerConfig(model_name=DEFAULT_MINIMAX_MODEL)
    return config


def _request_for_minimax(request: Request, fallback: ServerConfig) -> Request:
    """Return a request whose sampling config is valid for MiniMax.

    The caller's config object is left unchanged. A copy is attached only
    when the temperature has to be clamped, or when the request has no
    config of its own and the provider default needs clamping.
    """
    config = request.config or fallback
    clamped = _clamp_temperature(config.temperature)
    if clamped == config.temperature:
        return request
    return replace(request, config=replace(config, temperature=clamped))


class MiniMaxProvider(OpenAIProvider):
    """MiniMax API implementation of the Judge interface.

    Supported models include ``MiniMax-M3`` (default; 512K context, 128K
    max output, image input supported), ``MiniMax-M2.7``, and
    ``MiniMax-M2.7-highspeed``.

    Environment variables
    ---------------------
    MINIMAX_API_KEY : str
        API key for the MiniMax platform. ``API_TYPE=minimax`` selects this
        provider with no other configuration.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        # Skip OpenAIProvider.__init__: it builds a client from OPENAI_API_KEY,
        # which the documented MiniMax setup (`API_TYPE=minimax` plus
        # MINIMAX_API_KEY) does not set. Request handling still comes from
        # OpenAIProvider.evaluate.
        ServerInterface.__init__(self, _default_config(config))
        self.api_key = os.getenv("MINIMAX_API_KEY", "")
        self.base_url = MINIMAX_BASE_URL
        self.client = None
        self.use_client = False
        if not self.api_key:
            return
        # Deferred so a missing openai package still falls back to requests,
        # matching OpenAIProvider.
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_client = True
        except ImportError:
            eval_logger.warning("OpenAI client not available, falling back to requests for MiniMax")

    def evaluate(self, request: Request) -> Response:
        """Evaluate using the MiniMax API."""
        if not self.is_available():
            raise ValueError("MiniMax API key not configured (set MINIMAX_API_KEY)")
        response = super().evaluate(_request_for_minimax(request, self.config))
        response.content = _strip_think_tags(response.content)
        return response

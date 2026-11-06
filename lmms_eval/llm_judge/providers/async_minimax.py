"""Async MiniMax judge provider.

Reuses :class:`AsyncOpenAIProvider` with the same MiniMax defaults as
:class:`MiniMaxProvider`.
"""

import os
from typing import Any, Dict, Optional

from loguru import logger as eval_logger

from ..base import AsyncServerInterface
from ..protocol import Request, Response, ServerConfig
from .async_openai import AsyncOpenAIProvider
from .minimax import _default_config, _minimax_base_url, _minimax_extra_body, _request_for_minimax, _strip_think_tags


class AsyncMiniMaxProvider(AsyncOpenAIProvider):
    """Async MiniMax API implementation of the Judge interface."""

    def __init__(self, config: Optional[ServerConfig] = None):
        # Resolve config first: AsyncServerInterface reads max_concurrent from
        # the argument, so None crashes. Skip AsyncOpenAIProvider.__init__
        # because that client is built from OPENAI_API_KEY, which this
        # provider does not use. evaluate_async still comes from that class.
        config = _default_config(config)
        AsyncServerInterface.__init__(self, config)
        self.api_key = os.getenv("MINIMAX_API_KEY", "")
        self.base_url = _minimax_base_url()
        self.api_url = f"{self.base_url}/chat/completions"
        self.async_client = None
        self.use_async_client = False
        if not self.api_key:
            return
        # Deferred so a missing openai package still falls back to aiohttp,
        # matching AsyncOpenAIProvider.
        try:
            from openai import AsyncOpenAI

            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_async_client = True
        except ImportError:
            eval_logger.warning("AsyncOpenAI client not available, using aiohttp for MiniMax")

    def _extra_body(self, config: ServerConfig) -> Dict[str, Any]:
        return _minimax_extra_body(config)

    async def evaluate_async(self, request: Request) -> Response:
        """Evaluate using the MiniMax API asynchronously."""
        if not self.is_available():
            raise ValueError("MiniMax API key not configured (set MINIMAX_API_KEY)")
        response = await super().evaluate_async(_request_for_minimax(request, self.config))
        response.content = _strip_think_tags(response.content)
        return response

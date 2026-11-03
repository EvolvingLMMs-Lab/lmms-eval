"""Async TypeSafe Jev judge provider.

The HTTP call is the synchronous client in a worker thread. Payload
construction, retries, and the text adapter are shared with
:class:`JevProvider`, so async callers see the same decisions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ..base import AsyncServerInterface
from ..protocol import Request, Response, ServerConfig
from .jev import JevProvider, _default_config


class AsyncJevProvider(AsyncServerInterface):
    """Async TypeSafe Jev implementation of the judge interface.

    ``API_TYPE=async_jev`` and ``API_TYPE=async_typesafe`` both select this
    class. Credentials are ``TYPESAFE_API_KEY``, same as :class:`JevProvider`.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        # Resolve config first: AsyncServerInterface reads max_concurrent off
        # the argument, and None would crash before the Jev default applies.
        config = _default_config(config)
        super().__init__(config)
        self._sync = JevProvider(config)

    def is_available(self) -> bool:
        return self._sync.is_available()

    async def evaluate_async(self, request: Request) -> Response:
        """Evaluate one judgment without blocking the event loop."""
        async with self.semaphore:
            return await asyncio.to_thread(self._sync.evaluate, request)

    def evaluate_binary(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Binary noul judgment, including the raw yes-probability."""
        return self._sync.evaluate_binary(*args, **kwargs)

    def evaluate_with_rubric(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Per-item score judgment."""
        return self._sync.evaluate_with_rubric(*args, **kwargs)

    def evaluate_comparative(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Pairwise scores as two Jev score questions."""
        return self._sync.evaluate_comparative(*args, **kwargs)


__all__ = ["AsyncJevProvider"]

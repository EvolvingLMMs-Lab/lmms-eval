import abc
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .protocol import Request, Response, ServerConfig


class ServerInterface(abc.ABC):
    """Abstract base class for judge implementations"""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig(model_name="gpt-4")

    @abc.abstractmethod
    def evaluate(self, request: Request) -> Response:
        """
        Evaluate the given request and return a response

        Args:
            request: JudgeRequest containing the evaluation context

        Returns:
            JudgeResponse with the evaluation result
        """
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the judge service is available"""
        pass

    def prepare_messages(self, request: Request) -> List[Dict[str, Any]]:
        """Prepare messages in the format expected by the API"""
        messages = request.messages.copy()

        # Add system prompt if configured
        if self.config.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        return messages


class AsyncServerInterface(ServerInterface):
    """Abstract base class for async judge implementations"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    @abc.abstractmethod
    async def evaluate_async(self, request: Request) -> Response:
        """
        Asynchronously evaluate the given request and return a response

        Args:
            request: JudgeRequest containing the evaluation context

        Returns:
            JudgeResponse with the evaluation result
        """
        pass

    async def evaluate_batch(self, requests: List[Request]) -> List[Response]:
        """
        Evaluate multiple requests concurrently

        Args:
            requests: List of JudgeRequests to evaluate

        Returns:
            List of JudgeResponses in the same order as requests
        """
        tasks = [self.evaluate_async(request) for request in requests]
        return await asyncio.gather(*tasks)

    def evaluate(self, request: Request) -> Response:
        """Synchronous wrapper for async evaluation"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.evaluate_async(request))

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentModel(ABC):
    """Minimal single-request model interface."""

    @abstractmethod
    def generate(self, request: Any) -> Any:
        raise NotImplementedError


class ModelServer(ABC):
    """Model inference boundary for agent requests."""

    @abstractmethod
    def generate(self, request: Any) -> Any:
        raise NotImplementedError

    def generate_batch(self, requests: list[Any]) -> list[Any]:
        return [self.generate(request) for request in requests]

from __future__ import annotations

from abc import ABC, abstractmethod

from lmms_eval.agentic.types import AgentInput, AgentOutput


class AgentModel(ABC):
    """Minimal single-request model interface."""

    @abstractmethod
    def generate(self, request: AgentInput) -> AgentOutput:
        raise NotImplementedError


class ModelServer(ABC):
    """Batch/scheduling boundary for agent inference."""

    @abstractmethod
    def generate(self, request: AgentInput) -> AgentOutput:
        raise NotImplementedError

    def generate_batch(self, requests: list[AgentInput]) -> list[AgentOutput]:
        return [self.generate(request) for request in requests]

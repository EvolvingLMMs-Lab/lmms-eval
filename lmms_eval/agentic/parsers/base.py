from __future__ import annotations

from abc import ABC, abstractmethod

from lmms_eval.agentic.types import AgentInput, AgentOutput, EnvState, ParsedAction


class ModelOutputParser(ABC):
    """Raw model output -> normalized model output."""

    @abstractmethod
    def parse(self, output: AgentOutput, state: EnvState, agent_id: str | None = None) -> AgentOutput:
        raise NotImplementedError


class ActionParser(ABC):
    """Normalized model output -> game action."""

    @abstractmethod
    def parse(self, output: AgentOutput, state: EnvState, agent_id: str | None = None) -> ParsedAction:
        raise NotImplementedError


class ObservationParser(ABC):
    """Game state -> model input."""

    @abstractmethod
    def parse(self, state: EnvState, agent_id: str | None = None) -> AgentInput:
        raise NotImplementedError

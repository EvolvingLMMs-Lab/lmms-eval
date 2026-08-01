"""Typed parser interfaces shared by all agentic tasks and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lmms_eval.agentic.types import AgentInput, AgentOutput, EnvState, ParsedAction


@dataclass(slots=True)
class ParserContext:
    """Side-channel rollout state passed to every parser call."""

    state: EnvState | None = None
    agent_id: str | None = None
    step_idx: int | None = None
    request: AgentInput | None = None
    raw_output: AgentOutput | None = None
    history: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservationParser(ABC):
    """Environment state -> model request."""

    @abstractmethod
    def parse(self, state: EnvState, ctx: ParserContext) -> AgentInput:
        raise NotImplementedError


class ModelOutputParser(ABC):
    """Raw model output -> normalized model output."""

    @abstractmethod
    def parse(self, output: AgentOutput, ctx: ParserContext) -> AgentOutput:
        raise NotImplementedError


class ActionParser(ABC):
    """Normalized model output -> environment action."""

    @abstractmethod
    def parse(self, output: AgentOutput, ctx: ParserContext) -> ParsedAction:
        raise NotImplementedError

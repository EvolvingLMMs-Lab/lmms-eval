from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lmms_eval.agentic.types import ParsedAction


@dataclass(slots=True)
class ParserContext:
    """Runtime context for parser stages.

    Parser payloads are intentionally unconstrained. A parser can receive or
    return chat-style AgentInput/AgentOutput objects, tensors, arrays, custom
    policy dataclasses, or any other object. Context carries rollout state and
    side-channel information without baking those fields into every payload.
    """

    state: Any = None
    agent_id: str | None = None
    step_idx: int | None = None
    request: Any = None
    raw_output: Any = None
    history: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Parser(ABC):
    """Generic rollout parser stage: arbitrary object -> arbitrary object."""

    @abstractmethod
    def parse(self, value: Any, ctx: ParserContext) -> Any:
        raise NotImplementedError


class ModelOutputParser(Parser):
    """Raw model output -> normalized model output."""

    @abstractmethod
    def parse(self, value: Any, ctx: ParserContext) -> Any:
        raise NotImplementedError


class ActionParser(Parser):
    """Normalized model output -> game action."""

    @abstractmethod
    def parse(self, value: Any, ctx: ParserContext) -> ParsedAction:
        raise NotImplementedError


class ObservationParser(Parser):
    """Game state -> model input."""

    @abstractmethod
    def parse(self, value: Any, ctx: ParserContext) -> Any:
        raise NotImplementedError

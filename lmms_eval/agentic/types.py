from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContentBlock:
    """Open-ended model/env payload block.

    Text models can use ``text``/``image``/``video`` blocks; policy or JEPA-like
    models can use ``tensor``/``embedding``/``latent``/``logits`` without changing
    the framework.
    """

    type: str
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, text: str, **metadata: Any) -> "ContentBlock":
        return cls(type="text", data=text, metadata=dict(metadata))


@dataclass(slots=True)
class AgentInput:
    content: list[ContentBlock] = field(default_factory=list)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def first_text(self) -> str | None:
        for block in self.content:
            if block.type == "text" and isinstance(block.data, str):
                return block.data
        return None


@dataclass(slots=True)
class AgentOutput:
    content: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def first_text(self) -> str | None:
        for block in self.content:
            if block.type == "text" and isinstance(block.data, str):
                return block.data
        return None


@dataclass(slots=True)
class EnvState:
    env_id: str
    step_idx: int
    observation: Any
    terminal: bool = False
    active_agent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GameAction:
    type: str
    data: Any = None
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedAction:
    action: GameAction | dict[str, GameAction] | None = None
    is_submit: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    state: EnvState
    reward: float | dict[str, float] | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeStep:
    state: EnvState
    request: AgentInput | None = None
    raw_output: AgentOutput | None = None
    output: AgentOutput | None = None
    parsed_action: ParsedAction | None = None
    result: StepResult | None = None


@dataclass(slots=True)
class EpisodeResult:
    final_state: EnvState
    steps: list[EpisodeStep] = field(default_factory=list)
    success: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

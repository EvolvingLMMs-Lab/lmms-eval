"""Data types shared across the agentic game loop.

These dataclasses are the common vocabulary between environments, parsers,
model servers, and the episode loop. They have no dependencies beyond the
standard library, so every other module can import them freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContentBlock:
    """Open-ended model/env payload block.

    Text models use ``text``/``image``/``video`` blocks; policy or JEPA-like
    models can use ``tensor``/``embedding``/``latent``/``logits`` without
    changing the framework.
    """

    type: str
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, text: str, **metadata: Any) -> "ContentBlock":
        return cls(type="text", data=text, metadata=dict(metadata))


@dataclass(slots=True)
class AgentInput:
    """One model request: content blocks plus generation controls."""

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
    """One model response as content blocks."""

    content: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def first_text(self) -> str | None:
        for block in self.content:
            if block.type == "text" and isinstance(block.data, str):
                return block.data
        return None


@dataclass(slots=True)
class EnvState:
    """Environment snapshot handed to the observation parser."""

    env_id: str
    step_idx: int
    observation: Any
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GameAction:
    """Environment-facing action produced by the action parser."""

    type: str
    data: Any = None
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedAction:
    """Action-parser result: either an action or a parse error."""

    action: Any = None
    is_submit: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionDef:
    """One action an environment accepts.

    ``schema`` is a JSON-Schema-shaped ``{"properties": {...}, "required": [...]}``
    mapping for parameterized actions; ``None`` means the action takes no
    arguments.
    """

    name: str
    description: str | None = None
    schema: dict[str, Any] | None = None
    aliases: list[str] = field(default_factory=list)

    def render(self) -> str:
        signature = self.name
        properties = (self.schema or {}).get("properties") or {}
        if properties:
            rendered_params = ", ".join(f"{key}: {value.get('type', 'any')}" if isinstance(value, dict) else str(key) for key, value in properties.items())
            signature = f"{self.name}({rendered_params})"
        return f"{signature}: {self.description}" if self.description else signature


@dataclass(slots=True)
class ActionSpec:
    """Environment-declared action space (``EnvManager.action_spec()``).

    ``kind`` selects the default action parser and prompt rendering:
    ``discrete`` (pick one action name), ``parameterized`` (action name plus
    JSON arguments), or ``free_text`` (the whole model reply is the command
    and the environment validates it).
    """

    kind: str = "discrete"
    actions: list[ActionDef] = field(default_factory=list)
    submit_actions: list[str] = field(default_factory=list)
    prompt_hint: str | None = None

    def action_names(self) -> list[str]:
        return [action.name for action in self.actions]

    def alias_map(self) -> dict[str, str]:
        return {alias: action.name for action in self.actions for alias in action.aliases}

    def get(self, name: str) -> ActionDef | None:
        wanted = name.strip().upper()
        for action in self.actions:
            if action.name.upper() == wanted:
                return action
        return None

    def render_prompt(self) -> str:
        """Standard text for the ``{actions}`` prompt placeholder."""

        lines = [f"- {action.render()}" for action in self.actions]
        if self.prompt_hint:
            lines.append(self.prompt_hint)
        elif self.kind == "free_text" and not lines:
            lines.append("Respond with a single short text command.")
        return "\n".join(lines)


@dataclass(slots=True)
class StepResult:
    """Environment response to one action."""

    state: EnvState
    reward: float | dict[str, float] | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeStep:
    """Full record of one loop iteration."""

    state: EnvState
    request: AgentInput | None = None
    raw_output: AgentOutput | None = None
    output: AgentOutput | None = None
    parsed_action: ParsedAction | None = None
    result: StepResult | None = None


@dataclass(slots=True)
class EpisodeResult:
    """Terminal rollout summary consumed by the runner and task metrics."""

    final_state: EnvState
    steps: list[EpisodeStep] = field(default_factory=list)
    success: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

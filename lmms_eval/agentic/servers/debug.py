"""Deterministic server used for environment-loop smoke tests."""

from __future__ import annotations

from lmms_eval.agentic.servers.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock


class FixedActionModelServer(ModelServer):
    """Ignore the observation and always emit one configured action."""

    def __init__(self, action: str = "ATTACK") -> None:
        self.action = str(action).strip()

    def generate(self, request: AgentInput) -> AgentOutput:
        if not isinstance(request, AgentInput):
            raise TypeError(f"FixedActionModelServer requires AgentInput requests, got {type(request).__name__}")
        return AgentOutput(content=[ContentBlock.text(self.action)], metadata={"debug": True, "fixed_action": self.action})

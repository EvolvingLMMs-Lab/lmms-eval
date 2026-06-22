from __future__ import annotations

from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock


class FixedActionModelServer(ModelServer):
    """Debug model server that ignores the observation and always emits a fixed action.

    Useful for end-to-end smoke testing of the agentic game loop (env -> observation
    parser -> model server -> action parser -> env.step) without standing up a real
    VLM/vLLM backend. The default ``action="ATTACK"`` is parsed by ``VizDoomActionParser``
    into a single-button ATTACK action. Override via
    ``--agentic_model_server_args 'action=MOVE_LEFT'``.
    """

    def __init__(self, action: str = "ATTACK", **_: Any) -> None:
        self.action = str(action).strip()

    def generate(self, request: Any) -> AgentOutput:
        if not isinstance(request, AgentInput):
            raise TypeError(f"FixedActionModelServer requires AgentInput requests, got {type(request).__name__}")
        return AgentOutput(content=[ContentBlock.text(self.action)], metadata={"debug": True, "fixed_action": self.action})

    def generate_batch(self, requests: list[Any]) -> list[AgentOutput]:
        return [self.generate(request) for request in requests]

from __future__ import annotations

from lmms_eval.agentic.parsers.base import ModelOutputParser
from lmms_eval.agentic.types import AgentOutput, EnvState


class IdentityModelOutputParser(ModelOutputParser):
    """Pass model output through unchanged."""

    def parse(self, output: AgentOutput, state: EnvState, agent_id: str | None = None) -> AgentOutput:
        del state, agent_id
        return output

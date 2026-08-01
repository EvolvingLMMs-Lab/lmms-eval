"""Model-server interface for the agentic game loop."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lmms_eval.agentic.types import AgentInput, AgentOutput


class ModelServer(ABC):
    """Turn one agent request into one model response.

    Rollout workers may call ``generate`` concurrently, so concrete servers
    must be thread-safe.
    """

    def get_model_name(self) -> str | None:
        """Model identity used by task ``model_specific_parsers`` selection."""

        model = getattr(self, "model", None)
        return str(model) if model is not None else None

    @abstractmethod
    def generate(self, request: AgentInput) -> AgentOutput:
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_eval.agentic.types import EpisodeResult


class LoopWorker(ABC):
    """Base loop worker interface.

    Custom game loops can subclass this for multiplayer, turn-based,
    simultaneous-action, or environment-specific control flow.
    """

    @abstractmethod
    def run(self, doc: Any, seed: int | None = None, agent_id: str = "agent") -> EpisodeResult:
        raise NotImplementedError

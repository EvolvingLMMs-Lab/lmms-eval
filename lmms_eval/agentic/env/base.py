from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_eval.agentic.types import EnvState, GameAction, StepResult


class EnvManager(ABC):
    """Environment lifecycle and state boundary for agentic rollouts."""

    @abstractmethod
    def reset(self, doc: Any, seed: int | None = None) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: GameAction | dict[str, GameAction]) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> EnvState:
        raise NotImplementedError

    def close(self) -> None:
        return None

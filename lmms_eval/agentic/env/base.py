from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_eval.agentic.types import EnvState, GameAction, StepResult


class GameEnv(ABC):
    @abstractmethod
    def reset(self, doc: Any, seed: int | None = None) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: GameAction | dict[str, GameAction]) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class EnvManager(ABC):
    @abstractmethod
    def new_env(self, doc: Any, seed: int | None = None) -> EnvState | None:
        raise NotImplementedError

    @abstractmethod
    def step(self, env_id: str, action: GameAction | dict[str, GameAction]) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def get_state(self, env_id: str) -> EnvState:
        raise NotImplementedError

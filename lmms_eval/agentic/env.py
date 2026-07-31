"""Environment boundary for agentic rollouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_eval.agentic.types import EnvState, GameAction, StepResult


class EnvManager(ABC):
    """Gym-style environment lifecycle owned by one rollout.

    Implementations live next to the task that needs them (see
    ``lmms_eval/tasks/vizdoom_agentic/env.py``) and are referenced from the
    task YAML via ``game_env: !function utils.<factory>``.
    """

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

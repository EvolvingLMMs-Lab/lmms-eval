"""Environment boundary for agentic rollouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmms_eval.agentic.types import ActionSpec, EnvState, GameAction, StepResult


class EnvManager(ABC):
    """Gym-style environment lifecycle owned by one rollout.

    Concrete environments live next to their task (for example
    ``tasks/minigrid_agentic/env.py`` and ``tasks/vizdoom_agentic/env.py``).
    A task may reference one through a registry short name
    (``game_env: minigrid``) or a task-local factory.
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

    def action_spec(self) -> ActionSpec | None:
        """Declare the action space, when the environment can.

        Must be cheap and side-effect free; the loop may call it more than
        once per episode. Task-local parser functions can use the declaration
        both to render action help and to validate model output. ``None`` (the
        default) means the task parser owns those details without a shared
        declaration.
        """

        return None

    def close(self) -> None:
        return None

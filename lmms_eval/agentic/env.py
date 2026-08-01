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
        once per episode. Returning a spec lets the loop build a default
        action parser and render ``{actions}`` in observation templates.
        ``None`` (the default) means the task supplies its own action parser
        and prompt text.
        """

        return None

    def close(self) -> None:
        return None

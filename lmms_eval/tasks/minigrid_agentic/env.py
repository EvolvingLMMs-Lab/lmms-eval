"""Task-local MiniGrid environment for the agentic game loop.

The environment declares its action space via ``action_spec()``, emits the
reserved observation keys (``text`` / ``images`` / ``variables``), and keeps
simulator imports inside ``reset``. The task therefore needs no parser code:

    output_type: generate_until_game
    game_env: minigrid
    # observation_parser / action_parser omitted -> loop defaults

Dataset rows select the scenario (doc-as-scenario): ``env_id``, ``seed``, and
``max_episode_steps`` in the doc override the manager's defaults per episode.
Requires ``pip install gymnasium minigrid`` at rollout time only.
"""

from __future__ import annotations

from typing import Any

from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.types import (
    ActionDef,
    ActionSpec,
    EnvState,
    GameAction,
    StepResult,
)
from lmms_eval.imports import optional_import

_ACTION_INDICES = {"LEFT": 0, "RIGHT": 1, "FORWARD": 2, "PICKUP": 3, "DROP": 4, "TOGGLE": 5, "DONE": 6}

_ACTION_DEFS = [
    ActionDef(name="LEFT", description="turn left in place", aliases=["TURN_LEFT"]),
    ActionDef(name="RIGHT", description="turn right in place", aliases=["TURN_RIGHT"]),
    ActionDef(name="FORWARD", description="move one cell forward", aliases=["MOVE_FORWARD", "GO_FORWARD"]),
    ActionDef(name="PICKUP", description="pick up the object directly in front of you", aliases=["PICK_UP"]),
    ActionDef(name="DROP", description="drop the carried object on the cell in front of you"),
    ActionDef(name="TOGGLE", description="open or close the door (or interact with the object) in front of you", aliases=["OPEN", "USE"]),
    ActionDef(name="DONE", description="declare the mission complete"),
]

_DIRECTION_NAMES = {0: "east", 1: "south", 2: "west", 3: "north"}


class MiniGridEnvManager(EnvManager):
    """One MiniGrid episode rendered as full-grid RGB frames for a VLM policy."""

    def __init__(self, env_id: str = "MiniGrid-Empty-6x6-v0", max_episode_steps: int | None = None, actions: list[str] | None = None) -> None:
        self.env_id = env_id
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else None
        wanted = {name.upper() for name in actions} if actions else set(_ACTION_INDICES)
        unknown = wanted - set(_ACTION_INDICES)
        if unknown:
            raise ValueError(f"Unknown MiniGrid actions {sorted(unknown)}; expected a subset of {sorted(_ACTION_INDICES)}")
        self._actions = [action for action in _ACTION_DEFS if action.name in wanted]
        self._active_env_id = env_id
        self._env: Any = None
        self._mission = ""
        self._direction: int | None = None
        self._step_idx = 0
        self._invalid_actions = 0
        self._total_reward = 0.0
        self._terminal = False
        self._success = False

    def action_spec(self) -> ActionSpec:
        return ActionSpec(kind="discrete", actions=list(self._actions))

    def reset(self, doc: Any, seed: int | None = None) -> EnvState:
        doc = doc if isinstance(doc, dict) else {}
        gymnasium, has_gymnasium = optional_import("gymnasium")
        _, has_minigrid = optional_import("minigrid")
        if not (has_gymnasium and has_minigrid):
            raise ImportError("The minigrid environment requires `pip install gymnasium minigrid`.")

        self.close()
        self._active_env_id = str(doc.get("env_id") or self.env_id)
        max_episode_steps = doc.get("max_episode_steps", self.max_episode_steps)
        make_kwargs = {"render_mode": "rgb_array"}
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = int(max_episode_steps)
        self._env = gymnasium.make(self._active_env_id, **make_kwargs)

        if seed is None:
            seed = doc.get("seed")
        observation, _info = self._env.reset(seed=int(seed) if seed is not None else None)
        self._mission = str(observation.get("mission", "")) if isinstance(observation, dict) else ""
        self._direction = int(observation["direction"]) if isinstance(observation, dict) and "direction" in observation else None
        self._step_idx = 0
        self._invalid_actions = 0
        self._total_reward = 0.0
        self._terminal = False
        self._success = False
        return self.get_state()

    def step(self, action: GameAction | dict[str, GameAction]) -> StepResult:
        if self._env is None:
            raise RuntimeError("MiniGridEnvManager.step called before reset")
        if isinstance(action, dict):
            raise TypeError("MiniGridEnvManager is single-agent; got a per-agent action dict")

        self._step_idx += 1
        name = str(action.type).upper()
        reward = 0.0
        info: dict[str, Any] = {"requested_action": name}
        if name in _ACTION_INDICES and any(action_def.name == name for action_def in self._actions):
            observation, step_reward, terminated, truncated, step_info = self._env.step(_ACTION_INDICES[name])
            reward = float(step_reward)
            self._total_reward += reward
            self._terminal = bool(terminated or truncated)
            self._success = bool(terminated and reward > 0)
            if isinstance(observation, dict) and "direction" in observation:
                self._direction = int(observation["direction"])
            info.update({"terminated": bool(terminated), "truncated": bool(truncated), **(step_info or {})})
        else:
            # Invalid or unparsable action: burn the turn, leave the simulator untouched.
            self._invalid_actions += 1
            info["invalid_action"] = True

        state = self.get_state()
        return StepResult(state=state, reward=reward, done=state.terminal, info=info)

    def get_state(self) -> EnvState:
        observation = {
            "text": self._mission,
            "images": [self._env.render()] if self._env is not None else [],
            "variables": {"facing": _DIRECTION_NAMES.get(self._direction, str(self._direction)), "carrying": self._carrying()},
        }
        return EnvState(
            env_id=self._active_env_id,
            step_idx=self._step_idx,
            observation=observation,
            terminal=self._terminal,
            metadata={"success": self._success, "metrics": self._metrics()},
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _metrics(self) -> dict[str, float]:
        return {"steps": float(self._step_idx), "invalid_actions": float(self._invalid_actions), "reward": float(self._total_reward)}

    def _carrying(self) -> str | None:
        carrying = getattr(getattr(self._env, "unwrapped", self._env), "carrying", None)
        return getattr(carrying, "type", None) if carrying is not None else None

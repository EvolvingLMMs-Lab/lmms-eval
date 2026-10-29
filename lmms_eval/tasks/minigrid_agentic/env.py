"""Task-local MiniGrid simulator; optional dependencies load at episode reset."""

from __future__ import annotations

from typing import Any

from lmms_eval.imports import optional_import

ACTION_INDICES = {"LEFT": 0, "RIGHT": 1, "FORWARD": 2, "PICKUP": 3, "DROP": 4, "TOGGLE": 5, "DONE": 6}
ACTION_DESCRIPTIONS = {
    "LEFT": "turn left in place",
    "RIGHT": "turn right in place",
    "FORWARD": "move one cell forward",
    "PICKUP": "pick up the object directly in front of you",
    "DROP": "drop the carried object on the cell in front of you",
    "TOGGLE": "open or close the door in front of you",
    "DONE": "declare the mission complete",
}
DIRECTION_NAMES = {0: "east", 1: "south", 2: "west", 3: "north"}


class MiniGridEnvManager:
    """One MiniGrid episode rendered as a full-grid RGB image."""

    def __init__(self, env_id: str = "MiniGrid-Empty-6x6-v0", max_episode_steps: int | None = None) -> None:
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self._env: Any = None
        self._mission = ""
        self._direction = None
        self._step_idx = 0
        self._invalid_actions = 0
        self._total_reward = 0.0
        self._terminal = False
        self._success = False

    def reset(self, doc: dict, seed: int | None = None) -> dict:
        gymnasium, has_gymnasium = optional_import("gymnasium")
        _, has_minigrid = optional_import("minigrid")
        if not (has_gymnasium and has_minigrid):
            raise ImportError("The minigrid task requires gymnasium and minigrid; install with `uv add gymnasium minigrid`.")
        self.close()
        env_id = str(doc.get("env_id") or self.env_id)
        make_kwargs = {"render_mode": "rgb_array"}
        max_steps = doc.get("max_episode_steps", self.max_episode_steps)
        if max_steps is not None:
            make_kwargs["max_episode_steps"] = int(max_steps)
        self._env = gymnasium.make(env_id, **make_kwargs)
        seed = doc.get("seed") if seed is None else seed
        observation, _info = self._env.reset(seed=int(seed) if seed is not None else None)
        self._mission = str(observation.get("mission", ""))
        self._direction = observation.get("direction")
        self._step_idx = self._invalid_actions = 0
        self._total_reward = 0.0
        self._terminal = self._success = False
        return self.get_state()

    def step(self, action: str | None) -> dict:
        if self._env is None:
            raise RuntimeError("MiniGridEnvManager.step called before reset")
        self._step_idx += 1
        if action in ACTION_INDICES:
            observation, reward, terminated, truncated, _info = self._env.step(ACTION_INDICES[action])
            self._total_reward += float(reward)
            self._terminal = bool(terminated or truncated)
            self._success = bool(terminated and reward > 0)
            self._direction = observation.get("direction", self._direction)
        else:
            self._invalid_actions += 1
        return self.get_state()

    def get_state(self) -> dict:
        carrying = getattr(getattr(self._env, "unwrapped", self._env), "carrying", None)
        return {
            **self.summary(),
            "image": self._env.render() if self._env is not None else None,
            "variables": {"facing": DIRECTION_NAMES.get(self._direction, str(self._direction)), "carrying": getattr(carrying, "type", None)},
        }

    def summary(self) -> dict:
        return {
            "mission": self._mission,
            "terminal": self._terminal,
            "success": self._success,
            "metrics": {"steps": float(self._step_idx), "invalid_actions": float(self._invalid_actions), "reward": float(self._total_reward)},
        }

    def close(self) -> None:
        if self._env is not None:
            env, self._env = self._env, None
            env.close()

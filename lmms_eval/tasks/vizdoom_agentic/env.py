"""Task-local ViZDoom simulator, independent of the model backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lmms_eval.imports import optional_import

_ENUM_SETTINGS = {"mode": "Mode", "screen_format": "ScreenFormat", "screen_resolution": "ScreenResolution"}


class VizDoomEnv:
    def __init__(self, *, settings=None, tics_per_action=5, frame_history=5, success_reward_min=1.0):
        self.settings = dict(settings or {})
        self.tics_per_action = max(1, int(tics_per_action))
        self.frame_history = max(1, int(frame_history))
        self.success_reward_min = float(success_reward_min)
        self.game = None
        self.frames: list[Any] = []
        self.steps = 0
        self.invalid_actions = 0
        self.total_reward = 0.0
        self.terminal = False
        self.doc = {}
        self.buttons: list[str] = []

    def reset(self, doc, seed=None):
        self.close()
        vzd, available = optional_import("vizdoom")
        if not available:
            raise ImportError("ViZDoom evaluation requires `uv add vizdoom`.")
        self.doc = dict(doc)
        config = {**self.settings, **(doc.get("vizdoom") or {})}
        for key in ("config_path", "scenario_path", "game_path", "doom_map", "available_buttons"):
            if key in doc:
                config[key] = doc[key]
        self.game = vzd.DoomGame()
        config_path = config.pop("config_path", "basic.cfg")
        if config_path:
            self.game.load_config(str(_asset_path(config_path, vzd)))
        special = {"scenario_path": "doom_scenario_path", "game_path": "doom_game_path"}
        for key in ("audio_buffer", "notifications_buffer", "depth_buffer", "labels_buffer", "automap_buffer", "objects_info", "sectors_info"):
            special[key] = f"{key}_enabled"
        for key, value in config.items():
            if value is None:
                continue
            if key in _ENUM_SETTINGS:
                value = getattr(getattr(vzd, _ENUM_SETTINGS[key]), str(value).upper())
            elif key in {"available_buttons", "available_game_variables"}:
                enum = vzd.Button if key == "available_buttons" else vzd.GameVariable
                names = value.split(",") if isinstance(value, str) else value
                value = [getattr(enum, str(name).strip().upper()) for name in names]
            elif key in {"scenario_path", "game_path"}:
                value = str(_asset_path(value, vzd))
            setter = getattr(self.game, f"set_{special.get(key, key)}", None)
            if setter is None:
                raise ValueError(f"Unknown ViZDoom setting: {key}")
            setter(value)
        if seed is None:
            seed = doc.get("seed")
        if seed is not None:
            self.game.set_seed(int(seed))
        try:
            self.game.init()
            self.buttons = [button.name for button in self.game.get_available_buttons()]
            self.frames = []
            self.steps = self.invalid_actions = 0
            self.total_reward = 0.0
            self.terminal = bool(self.game.is_episode_finished())
            self._capture_frame()
            return self.observation()
        except BaseException:
            self.close()
            raise

    def step(self, action):
        if self.game is None:
            raise RuntimeError("ViZDoom step called before reset")
        if self.terminal:
            return self.observation()
        self.steps += 1
        if action.get("submit"):
            self.terminal = True
            return self.observation()
        buttons = action.get("buttons")
        valid = isinstance(buttons, dict) and all(name in self.buttons for name in buttons)
        if not valid:
            self.invalid_actions += 1
            buttons = {}
        vector = [float(buttons.get(name, 0)) for name in self.buttons]
        # Each decision advances exactly five tics by default. Keep a frame from
        # every tic, so the next request sees the clip produced by this action.
        tics = max(1, min(35, int(action.get("tics", self.tics_per_action))))
        for _ in range(tics):
            if self.game.is_episode_finished():
                break
            self.total_reward += float(self.game.make_action(vector, 1))
            self._capture_frame()
        self.terminal = bool(self.game.is_episode_finished())
        return self.observation()

    def observation(self):
        return {
            "instruction": self.doc.get("instruction", ""),
            "available_buttons": list(self.buttons),
            "decision_tics": self.tics_per_action,
            "frames": list(self.frames),
            "terminal": self.terminal,
            "metrics": self.metrics(),
        }

    def metrics(self):
        return {
            "vizdoom_success": float(self.total_reward >= self.success_reward_min),
            "vizdoom_steps": float(self.steps),
            "vizdoom_invalid_actions": float(self.invalid_actions),
            "vizdoom_total_reward": self.total_reward,
        }

    def close(self):
        game, self.game = self.game, None
        if game is not None:
            game.close()

    def _capture_frame(self):
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            self.frames.append(state.screen_buffer.copy())
            self.frames = self.frames[-self.frame_history :]


def _asset_path(value, vzd):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    for root in (Path(vzd.scenarios_path), Path(__file__).parent):
        candidate = root / path
        if candidate.exists():
            return candidate
    return path

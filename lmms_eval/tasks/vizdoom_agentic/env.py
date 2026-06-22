from __future__ import annotations

from typing import Any

from lmms_eval.agentic.env.base import EnvManager
from lmms_eval.agentic.types import EnvState, GameAction, StepResult
from lmms_eval.imports import optional_import

_BUFFER_KEYS = {
    "screen_buffer",
    "depth_buffer",
    "labels_buffer",
    "automap_buffer",
    "audio_buffer",
    "notifications_buffer",
}
_ENUM_SETTING_TYPES = {
    "mode": "Mode",
    "screen_format": "ScreenFormat",
    "screen_resolution": "ScreenResolution",
    "audio_sampling_rate": "SamplingRate",
    "automap_mode": "AutomapMode",
}


class VizDoomEnvManager(EnvManager):
    """ViZDoom-backed single-agent environment manager.

    The environment exposes VizDoom's native buffers and metadata through
    ``EnvState.observation``. It is intentionally model-agnostic: model-side
    serving and output cleanup remain CLI/runtime choices.
    """

    def __init__(
        self,
        config_path: str | None = "basic.cfg",
        scenario_path: str | None = None,
        game_path: str | None = None,
        doom_map: str | None = None,
        mode: str | None = "PLAYER",
        available_buttons: list[str] | str | None = None,
        available_game_variables: list[str] | str | None = None,
        button_max_values: dict[str, float] | None = None,
        tracked_game_variables: list[str] | str | None = None,
        screen_resolution: str | None = None,
        screen_format: str | None = None,
        depth_buffer: bool | None = None,
        labels_buffer: bool | None = None,
        automap_buffer: bool | None = None,
        automap_mode: str | None = None,
        automap_render_textures: bool | None = None,
        automap_render_objects_as_sprites: bool | None = None,
        automap_rotate: bool | None = None,
        objects_info: bool | None = None,
        sectors_info: bool | None = None,
        audio_buffer: bool | None = None,
        audio_buffer_size: int | None = None,
        audio_sampling_rate: str | None = None,
        notifications_buffer: bool | None = None,
        notifications_buffer_size: int | None = None,
        sound_enabled: bool | None = None,
        window_visible: bool | None = False,
        render_hud: bool | None = None,
        render_minimal_hud: bool | None = None,
        render_crosshair: bool | None = None,
        render_weapon: bool | None = None,
        render_decals: bool | None = None,
        render_particles: bool | None = None,
        render_effects_sprites: bool | None = None,
        render_messages: bool | None = None,
        render_corpses: bool | None = None,
        render_screen_flashes: bool | None = None,
        render_all_frames: bool | None = None,
        game_args: str | list[str] | None = None,
        doom_skill: int | None = None,
        episode_timeout: int | None = None,
        episode_start_time: int | None = None,
        ticrate: int | None = None,
        tics_per_action: int = 1,
        capture_action_frames: bool | str = False,
        frame_history: int = 1,
        success_reward_min: float | None = None,
        success_variable: str | None = None,
        success_variable_min: float | None = None,
        max_labels: int | None = 256,
        max_objects: int | None = 256,
        max_sectors: int | None = 128,
        settings: dict[str, Any] | None = None,
        doc: Any = None,
        lmms_eval_specific_kwargs: dict[str, Any] | None = None,
        **extra_settings: Any,
    ) -> None:
        del doc, lmms_eval_specific_kwargs
        self.config = {
            "config_path": config_path,
            "scenario_path": scenario_path,
            "game_path": game_path,
            "doom_map": doom_map,
            "mode": mode,
            "available_buttons": _as_list(available_buttons),
            "available_game_variables": _as_list(available_game_variables),
            "button_max_values": dict(button_max_values or {}),
            "screen_resolution": screen_resolution,
            "screen_format": screen_format,
            "depth_buffer": depth_buffer,
            "labels_buffer": labels_buffer,
            "automap_buffer": automap_buffer,
            "automap_mode": automap_mode,
            "automap_render_textures": automap_render_textures,
            "automap_render_objects_as_sprites": automap_render_objects_as_sprites,
            "automap_rotate": automap_rotate,
            "objects_info": objects_info,
            "sectors_info": sectors_info,
            "audio_buffer": audio_buffer,
            "audio_buffer_size": audio_buffer_size,
            "audio_sampling_rate": audio_sampling_rate,
            "notifications_buffer": notifications_buffer,
            "notifications_buffer_size": notifications_buffer_size,
            "sound_enabled": sound_enabled,
            "window_visible": window_visible,
            "render_hud": render_hud,
            "render_minimal_hud": render_minimal_hud,
            "render_crosshair": render_crosshair,
            "render_weapon": render_weapon,
            "render_decals": render_decals,
            "render_particles": render_particles,
            "render_effects_sprites": render_effects_sprites,
            "render_messages": render_messages,
            "render_corpses": render_corpses,
            "render_screen_flashes": render_screen_flashes,
            "render_all_frames": render_all_frames,
            "game_args": game_args,
            "doom_skill": doom_skill,
            "episode_timeout": episode_timeout,
            "episode_start_time": episode_start_time,
            "ticrate": ticrate,
        }
        self.tracked_game_variables = _as_list(tracked_game_variables)
        self.tics_per_action = max(1, int(tics_per_action))
        self.capture_action_frames = _as_bool(capture_action_frames)
        self.frame_history = max(1, int(frame_history))
        self.success_reward_min = success_reward_min
        self.success_variable = success_variable
        self.success_variable_min = success_variable_min
        self.max_labels = max_labels
        self.max_objects = max_objects
        self.max_sectors = max_sectors
        self.settings = {**(settings or {}), **extra_settings}

        self.vzd = None
        self.game = None
        self.doc: dict[str, Any] = {}
        self.env_id = "vizdoom"
        self.step_idx = 0
        self.invalid_actions = 0
        self.last_reward = 0.0
        self.last_observation: dict[str, Any] = {}
        self.screen_history: list[Any] = []
        self.terminal_override = False

    def reset(self, doc: Any, seed: int | None = None) -> EnvState:
        self.close()
        self.vzd = _require_vizdoom()
        self.doc = dict(doc or {})
        self.env_id = str(self.doc.get("id", self.doc.get("scenario", "vizdoom")))
        self.step_idx = 0
        self.invalid_actions = 0
        self.last_reward = 0.0
        self.last_observation = {}
        self.screen_history = []
        self.terminal_override = False

        config = self._merged_config(self.doc)
        self.game = self.vzd.DoomGame()
        self._configure_game(self.game, config, seed=seed)
        self.game.init()
        return self._state()

    def step(self, action: GameAction | dict[str, GameAction]) -> StepResult:
        if self.game is None:
            raise RuntimeError("VizDoomEnvManager.step() called before reset()")
        if isinstance(action, dict):
            action = next(iter(action.values()))

        if self.terminal_override or self.game.is_episode_finished():
            return StepResult(state=self._state(), reward=0.0, done=True, info={"already_terminal": True})

        action_vector, tics, action_info, valid = self._action_to_vector(action)
        if action.type.upper() == "SUBMIT":
            self.terminal_override = True
            self.last_reward = 0.0
            return StepResult(state=self._state(), reward=0.0, done=True, info={"action": "SUBMIT"})

        if not valid:
            self.invalid_actions += 1

        if self.capture_action_frames and tics > 1:
            self.last_reward, elapsed_tics, captured_frames = self._make_action_and_capture_frames(action_vector, tics)
            self.step_idx += 1
            state = self._state(capture_screen_frame=False)
        else:
            self.last_reward = float(self.game.make_action(action_vector, tics))
            elapsed_tics = tics
            captured_frames = 0
            self.step_idx += 1
            state = self._state()
        info = {
            **action_info,
            "action_vector": action_vector,
            "tics": tics,
            "elapsed_tics": elapsed_tics,
            "captured_action_frames": captured_frames,
            "last_action": _safe_sequence(self.game.get_last_action()),
            "total_reward": float(self.game.get_total_reward()),
            "invalid_actions": self.invalid_actions,
            "episode_finished": bool(self.game.is_episode_finished()),
            "timeout": _safe_call(self.game.is_episode_timeout_reached, default=False),
            "player_dead": _safe_call(self.game.is_player_dead, default=False),
        }
        return StepResult(state=state, reward=self.last_reward, done=state.terminal, info=info)

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None

    def get_state(self) -> EnvState:
        return self._state(capture_screen_frame=False)

    def _merged_config(self, doc: dict[str, Any]) -> dict[str, Any]:
        doc_config = doc.get("vizdoom", {})
        if not isinstance(doc_config, dict):
            doc_config = {}
        config = dict(self.config)
        config.update(doc_config)
        for key in ["config_path", "scenario_path", "game_path", "doom_map", "available_buttons", "available_game_variables"]:
            if key in doc:
                config[key] = doc[key]
        config["available_buttons"] = _as_list(config.get("available_buttons"))
        config["available_game_variables"] = _as_list(config.get("available_game_variables"))
        return config

    def _configure_game(self, game: Any, config: dict[str, Any], seed: int | None) -> None:
        if config.get("config_path"):
            game.load_config(str(config["config_path"]))
        if config.get("scenario_path"):
            game.set_doom_scenario_path(str(config["scenario_path"]))
        if config.get("game_path"):
            game.set_doom_game_path(str(config["game_path"]))
        if config.get("doom_map"):
            game.set_doom_map(str(config["doom_map"]))
        if seed is not None:
            game.set_seed(int(seed))

        self._apply_enum_setting(game, "mode", config.get("mode"))
        self._apply_enum_setting(game, "screen_resolution", config.get("screen_resolution"))
        self._apply_enum_setting(game, "screen_format", config.get("screen_format"))
        self._apply_enum_setting(game, "audio_sampling_rate", config.get("audio_sampling_rate"))
        self._apply_enum_setting(game, "automap_mode", config.get("automap_mode"))

        if config.get("available_buttons"):
            game.set_available_buttons([_enum_value(self.vzd, "Button", name) for name in config["available_buttons"]])
        if config.get("available_game_variables"):
            game.set_available_game_variables([_enum_value(self.vzd, "GameVariable", name) for name in config["available_game_variables"]])
        for button, max_value in dict(config.get("button_max_values") or {}).items():
            game.set_button_max_value(_enum_value(self.vzd, "Button", button), float(max_value))

        direct_settings = {
            "depth_buffer_enabled": config.get("depth_buffer"),
            "labels_buffer_enabled": config.get("labels_buffer"),
            "automap_buffer_enabled": config.get("automap_buffer"),
            "automap_render_textures": config.get("automap_render_textures"),
            "automap_render_objects_as_sprites": config.get("automap_render_objects_as_sprites"),
            "automap_rotate": config.get("automap_rotate"),
            "objects_info_enabled": config.get("objects_info"),
            "sectors_info_enabled": config.get("sectors_info"),
            "audio_buffer_enabled": config.get("audio_buffer"),
            "audio_buffer_size": config.get("audio_buffer_size"),
            "notifications_buffer_enabled": config.get("notifications_buffer"),
            "notifications_buffer_size": config.get("notifications_buffer_size"),
            "sound_enabled": config.get("sound_enabled"),
            "window_visible": config.get("window_visible"),
            "render_hud": config.get("render_hud"),
            "render_minimal_hud": config.get("render_minimal_hud"),
            "render_crosshair": config.get("render_crosshair"),
            "render_weapon": config.get("render_weapon"),
            "render_decals": config.get("render_decals"),
            "render_particles": config.get("render_particles"),
            "render_effects_sprites": config.get("render_effects_sprites"),
            "render_messages": config.get("render_messages"),
            "render_corpses": config.get("render_corpses"),
            "render_screen_flashes": config.get("render_screen_flashes"),
            "render_all_frames": config.get("render_all_frames"),
            "doom_skill": config.get("doom_skill"),
            "episode_timeout": config.get("episode_timeout"),
            "episode_start_time": config.get("episode_start_time"),
            "ticrate": config.get("ticrate"),
        }
        for key, value in direct_settings.items():
            self._apply_setting(game, key, value)

        game_args = config.get("game_args")
        if isinstance(game_args, list):
            for args in game_args:
                game.add_game_args(str(args))
        elif game_args:
            game.set_game_args(str(game_args))

        for key, value in self.settings.items():
            self._apply_setting(game, key, value)

    def _apply_enum_setting(self, game: Any, key: str, value: Any) -> None:
        if value is None:
            return
        enum_type = _ENUM_SETTING_TYPES[key]
        self._apply_setting(game, key, _enum_value(self.vzd, enum_type, value))

    @staticmethod
    def _apply_setting(game: Any, key: str, value: Any) -> None:
        if value is None:
            return
        method_name = key if key.startswith("set_") else f"set_{key}"
        method = getattr(game, method_name, None)
        if method is None:
            raise KeyError(f"Unknown VizDoom setting '{key}'")
        method(value)

    def _state(self, *, capture_screen_frame: bool = True) -> EnvState:
        if self.game is None:
            raise RuntimeError("VizDoomEnvManager state requested before reset()")
        terminal = self.terminal_override or bool(self.game.is_episode_finished())
        state = None if terminal else self.game.get_state()
        observation = self._observation_from_state(state, capture_screen_frame=capture_screen_frame)
        self.last_observation = observation
        metrics = self._metrics(observation, terminal=terminal)
        success = self._success(observation, metrics, terminal=terminal)
        if success is not None:
            metrics["vizdoom_success"] = 1.0 if success else 0.0
        return EnvState(
            env_id=self.env_id,
            step_idx=self.step_idx,
            observation=observation,
            active_agent_ids=[] if terminal else ["agent"],
            terminal=terminal,
            metadata={"success": success, "metrics": metrics},
        )

    def _observation_from_state(self, state: Any, *, capture_screen_frame: bool = True) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "instruction": self.doc.get("instruction", ""),
            "env_id": self.env_id,
            "step_idx": self.step_idx,
            "available_buttons": self._available_button_names(),
            "available_game_variables": self._available_game_variable_names(),
            "screen": self._screen_info(),
            "decision_tics": self.tics_per_action,
            "screen_history_length": len(self.screen_history),
            "episode_time": _safe_call(self.game.get_episode_time, default=0),
            "total_reward": float(_safe_call(self.game.get_total_reward, default=0.0)),
            "last_reward": self.last_reward,
            "last_action": _safe_sequence(_safe_call(self.game.get_last_action, default=[])),
        }
        if state is None:
            observation.update({key: value for key, value in self.last_observation.items() if key in _BUFFER_KEYS or key == "screen_history"})
            return observation

        observation["state_number"] = getattr(state, "number", None)
        observation["tic"] = getattr(state, "tic", None)
        observation["game_variables"] = self._game_variables(state)
        observation["tracked_game_variables"] = self._tracked_game_variables()

        for attr in _BUFFER_KEYS:
            value = getattr(state, attr, None)
            if value is not None:
                observation[attr] = value
                if attr == "screen_buffer":
                    if capture_screen_frame:
                        self._append_screen_frame(value)
                    observation["screen_history"] = list(self.screen_history)

        labels = getattr(state, "labels", None)
        if labels is not None:
            observation["labels"] = [_label_to_dict(label) for label in _take(labels, self.max_labels)]
        objects = getattr(state, "objects", None)
        if objects is not None:
            observation["objects"] = [_object_to_dict(obj) for obj in _take(objects, self.max_objects)]
        sectors = getattr(state, "sectors", None)
        if sectors is not None:
            observation["sectors"] = [_sector_to_dict(sector) for sector in _take(sectors, self.max_sectors)]
        server_state = _safe_call(self.game.get_server_state, default=None)
        if server_state is not None:
            observation["server_state"] = _server_state_to_dict(server_state)
        return observation

    def _make_action_and_capture_frames(self, action_vector: list[float], tics: int) -> tuple[float, int, int]:
        reward = 0.0
        elapsed_tics = 0
        captured_frames = 0
        for _ in range(tics):
            if self.game.is_episode_finished():
                break
            reward += float(self.game.make_action(action_vector, 1))
            elapsed_tics += 1
            if self.game.is_episode_finished():
                break
            state = self.game.get_state()
            frame = getattr(state, "screen_buffer", None) if state is not None else None
            if frame is not None:
                self._append_screen_frame(frame)
                captured_frames += 1
        return reward, elapsed_tics, captured_frames

    def _append_screen_frame(self, frame: Any) -> None:
        if hasattr(frame, "copy"):
            frame = frame.copy()
        self.screen_history.append(frame)
        if len(self.screen_history) > self.frame_history:
            self.screen_history = self.screen_history[-self.frame_history :]

    def _game_variables(self, state: Any) -> dict[str, float]:
        names = self._available_game_variable_names()
        values = getattr(state, "game_variables", None)
        if values is None:
            return {}
        return {name: float(value) for name, value in zip(names, values, strict=False)}

    def _tracked_game_variables(self) -> dict[str, float]:
        tracked = {}
        for name in self.tracked_game_variables:
            tracked[name] = float(self.game.get_game_variable(_enum_value(self.vzd, "GameVariable", name)))
        return tracked

    def _metrics(self, observation: dict[str, Any], terminal: bool) -> dict[str, float]:
        del observation
        return {
            "vizdoom_reward": float(self.last_reward),
            "vizdoom_total_reward": float(_safe_call(self.game.get_total_reward, default=0.0)),
            "vizdoom_steps": float(self.step_idx),
            "vizdoom_invalid_actions": float(self.invalid_actions),
            "vizdoom_episode_finished": 1.0 if terminal else 0.0,
            "vizdoom_timeout": 1.0 if _safe_call(self.game.is_episode_timeout_reached, default=False) else 0.0,
            "vizdoom_player_dead": 1.0 if _safe_call(self.game.is_player_dead, default=False) else 0.0,
        }

    def _success(self, observation: dict[str, Any], metrics: dict[str, float], terminal: bool) -> bool | None:
        if self.success_reward_min is not None:
            return metrics["vizdoom_total_reward"] >= float(self.success_reward_min)
        if self.success_variable:
            values = observation.get("game_variables", {}) | observation.get("tracked_game_variables", {})
            if self.success_variable not in values:
                return False
            threshold = 1.0 if self.success_variable_min is None else float(self.success_variable_min)
            return float(values[self.success_variable]) >= threshold
        if terminal:
            return not bool(_safe_call(self.game.is_episode_timeout_reached, default=False) or _safe_call(self.game.is_player_dead, default=False))
        return False

    def _available_button_names(self) -> list[str]:
        return [button.name for button in self.game.get_available_buttons()]

    def _available_game_variable_names(self) -> list[str]:
        return [variable.name for variable in self.game.get_available_game_variables()]

    def _screen_info(self) -> dict[str, Any]:
        return {
            "width": int(_safe_call(self.game.get_screen_width, default=0)),
            "height": int(_safe_call(self.game.get_screen_height, default=0)),
            "channels": int(_safe_call(self.game.get_screen_channels, default=0)),
            "format": _enum_name(_safe_call(self.game.get_screen_format, default=None)),
        }

    def _action_to_vector(self, action: GameAction) -> tuple[list[float], int, dict[str, Any], bool]:
        available = self._available_button_names()
        vector = [0.0] * len(available)
        action_type = action.type.upper()
        data = action.data
        tics = self._action_tics(action)
        info = {"action": action.type, "buttons": {}}

        if action_type == "PARSE_ERROR":
            info["error"] = data
            return vector, tics, info, False
        if action_type == "NOOP":
            return vector, tics, info, True
        if action_type in {"VIZDOOM_ACTION", "VIZDOOM_BUTTONS", "BUTTON_VECTOR", "COMBO"}:
            valid = self._apply_action_data(vector, available, data, info)
            return vector, tics, info, valid
        if action_type in available:
            value = float(data) if isinstance(data, int | float) else 1.0
            vector[available.index(action_type)] = value
            info["buttons"][action_type] = value
            return vector, tics, info, True
        info["error"] = f"button is not available in this VizDoom scenario: {action.type}"
        return vector, tics, info, False

    def _apply_action_data(self, vector: list[float], available: list[str], data: Any, info: dict[str, Any]) -> bool:
        if isinstance(data, list):
            if len(data) == len(available) and all(isinstance(value, int | float | bool) for value in data):
                for idx, value in enumerate(data):
                    vector[idx] = float(value)
                info["buttons"] = {name: value for name, value in zip(available, vector, strict=False) if value}
                return True
            return self._apply_button_values(vector, available, {name: 1.0 for name in data}, info)
        if isinstance(data, dict):
            if isinstance(data.get("values"), list):
                return self._apply_action_data(vector, available, data["values"], info)
            buttons = data.get("buttons", data.get("button_values", data.get("actions", data.get("action"))))
            if isinstance(buttons, str):
                buttons = [buttons]
            if isinstance(buttons, list):
                return self._apply_button_values(vector, available, {name: 1.0 for name in buttons}, info)
            if isinstance(buttons, dict):
                return self._apply_button_values(vector, available, buttons, info)
        info["error"] = f"unsupported VizDoom action payload: {data!r}"
        return False

    @staticmethod
    def _apply_button_values(vector: list[float], available: list[str], buttons: dict[str, Any], info: dict[str, Any]) -> bool:
        valid = True
        normalized_buttons = {}
        for name, value in buttons.items():
            button_name = str(name).upper()
            if button_name not in available:
                valid = False
                continue
            numeric_value = float(value)
            vector[available.index(button_name)] = numeric_value
            normalized_buttons[button_name] = numeric_value
        info["buttons"] = normalized_buttons
        if not valid:
            info["error"] = "one or more buttons are not available in this VizDoom scenario"
        return valid

    def _action_tics(self, action: GameAction) -> int:
        if isinstance(action.data, dict) and "tics" in action.data:
            return max(1, int(action.data["tics"]))
        if "tics" in action.metadata:
            return max(1, int(action.metadata["tics"]))
        return self.tics_per_action


VizDoomEnv = VizDoomEnvManager


def _require_vizdoom():
    vizdoom, has_vizdoom = optional_import("vizdoom")
    if not has_vizdoom:
        raise ImportError("The 'vizdoom' package is required for the VizDoom EnvManager. Install it with `pip install vizdoom`.")
    return vizdoom


def _enum_value(vzd: Any, enum_type: str, value: Any) -> Any:
    enum_cls = getattr(vzd, enum_type)
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, int):
        return enum_cls(value)
    name = str(value).upper()
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:
        available = ", ".join(getattr(enum_cls, "__members__", {}).keys())
        raise KeyError(f"Unknown VizDoom {enum_type} '{value}'. Available: {available}") from exc


def _enum_name(value: Any) -> str | None:
    return getattr(value, "name", None) if value is not None else None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    return [str(item).upper() for item in value]


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_call(func: Any, default: Any = None) -> Any:
    try:
        return func()
    except Exception:
        return default


def _safe_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    return [float(item) if isinstance(item, int | float) else item for item in value]


def _take(values: Any, limit: int | None) -> list[Any]:
    items = list(values)
    return items if limit is None else items[: int(limit)]


def _label_to_dict(label: Any) -> dict[str, Any]:
    return {
        "value": label.value,
        "x": label.x,
        "y": label.y,
        "width": label.width,
        "height": label.height,
        "object_id": label.object_id,
        "object_name": label.object_name,
        "object_category": label.object_category,
        "object_position": [label.object_position_x, label.object_position_y, label.object_position_z],
        "object_velocity": [label.object_velocity_x, label.object_velocity_y, label.object_velocity_z],
        "object_angle": label.object_angle,
        "object_pitch": label.object_pitch,
        "object_roll": label.object_roll,
    }


def _object_to_dict(obj: Any) -> dict[str, Any]:
    return {
        "id": obj.id,
        "name": obj.name,
        "position": [obj.position_x, obj.position_y, obj.position_z],
        "velocity": [obj.velocity_x, obj.velocity_y, obj.velocity_z],
        "angle": obj.angle,
        "pitch": obj.pitch,
        "roll": obj.roll,
    }


def _line_to_dict(line: Any) -> dict[str, Any]:
    return {"x1": line.x1, "y1": line.y1, "x2": line.x2, "y2": line.y2, "is_blocking": line.is_blocking}


def _sector_to_dict(sector: Any) -> dict[str, Any]:
    return {
        "floor_height": sector.floor_height,
        "ceiling_height": sector.ceiling_height,
        "lines": [_line_to_dict(line) for line in sector.lines],
    }


def _server_state_to_dict(server_state: Any) -> dict[str, Any]:
    return {
        "tic": server_state.tic,
        "player_count": server_state.player_count,
        "players_in_game": list(server_state.players_in_game),
        "players_names": list(server_state.players_names),
        "players_frags": list(server_state.players_frags),
        "players_afk": list(server_state.players_afk),
        "players_last_action_tic": list(server_state.players_last_action_tic),
        "players_last_kill_tic": list(server_state.players_last_kill_tic),
    }

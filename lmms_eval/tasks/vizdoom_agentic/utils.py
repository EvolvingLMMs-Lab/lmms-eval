"""ViZDoom callbacks for the existing generate_until_agentic loop."""

import importlib.util
import json
import os
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path


def vizdoom_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return str(doc.get("instruction", "Choose the next ViZDoom action."))


def vizdoom_doc_to_messages(doc, *, round_idx=None, previous_output=None, previous_round_info=None, lmms_eval_specific_kwargs=None):
    parsers = _sibling_module("parsers")
    if round_idx is None:
        # Request building must not start a simulator or retain a live handle.
        return [{"role": "user", "content": [{"type": "text", "text": doc.get("instruction", "")}]}]
    info = previous_round_info
    if info is None:
        env = vizdoom_env_manager(lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
        info = {"env": env, "close": env.close}
        try:
            observation = env.reset(doc)
        except BaseException:
            env.close()
            raise
        info["finalize"] = lambda: _final_response(env)
    else:
        env = info["env"]
        action = parsers.parse_vizdoom_action(previous_output[-1] if previous_output else "", env.buttons)
        observation = env.step(action)
    info["state"] = {"metrics": observation["metrics"], "terminal": observation["terminal"]}
    if observation["terminal"]:
        return None, True, [*(previous_output or []), _final_response(env)], info
    try:
        messages = parsers.vizdoom_messages(observation)
    except BaseException:
        env.close()
        raise
    return messages, False, previous_output, info


def _final_response(env):
    metrics = env.metrics()
    return json.dumps({"success": bool(metrics["vizdoom_success"]), "metrics": metrics})


def vizdoom_doc_to_target(doc):
    return "maximize_reward"


def vizdoom_env_manager(doc=None, lmms_eval_specific_kwargs=None):
    del doc
    kwargs = lmms_eval_specific_kwargs or {}
    settings = {
        "config_path": "basic.cfg",
        "doom_config_path": _vizdoom_runtime_config_path(),
        "mode": "PLAYER",
        "screen_resolution": "RES_320X240",
        "screen_format": "RGB24",
        "available_buttons": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        "render_hud": True,
        "render_weapon": True,
        "render_messages": True,
        "render_screen_flashes": True,
        "render_particles": True,
        "render_decals": True,
        "render_corpses": True,
        "render_effects_sprites": True,
        "render_crosshair": False,
        "depth_buffer_enabled": False,
        "labels_buffer_enabled": False,
        "automap_buffer_enabled": False,
        "objects_info_enabled": False,
        "sectors_info_enabled": False,
        "audio_buffer_enabled": False,
        "sound_enabled": False,
        "window_visible": False,
        **kwargs.get("settings", {}),
    }
    return _sibling_module("env").VizDoomEnv(settings=settings, tics_per_action=kwargs.get("tics_per_action", 5), frame_history=kwargs.get("frame_history", 5), success_reward_min=kwargs.get("success_reward_min", 1.0))


@lru_cache(maxsize=None)
def _sibling_module(name):
    module_path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"lmms_eval_vizdoom_agentic_{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vizdoom_runtime_config_path():
    root = Path(os.getenv("VIZDOOM_CONFIG_DIR", tempfile.gettempdir())) / "lmms_eval_vizdoom"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"_vizdoom_{os.getpid()}_{uuid.uuid4().hex}.ini")


def vizdoom_process_results(doc, results):
    try:
        payload = json.loads(results[0] if results else "{}")
    except (json.JSONDecodeError, TypeError):
        payload = {}
    metrics = payload.get("metrics", {})
    return {
        "vizdoom_success": float(metrics.get("vizdoom_success", 0.0)),
        "vizdoom_steps": float(metrics.get("vizdoom_steps", 0.0)),
        "vizdoom_invalid_actions": float(metrics.get("vizdoom_invalid_actions", 0.0)),
        "vizdoom_env_error": float(metrics.get("env_error", payload.get("env_error", 0.0))),
    }


def vizdoom_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0

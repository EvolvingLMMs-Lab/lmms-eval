import importlib.util
import json
import os
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path


def vizdoom_doc_to_visual(doc):
    return []


def vizdoom_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['instruction']}\nUse the VizDoom visual input and state to choose the next action.{post_prompt}"


def vizdoom_doc_to_target(doc):
    return "maximize_reward"


def vizdoom_env_manager(doc=None, lmms_eval_specific_kwargs=None):
    del doc, lmms_eval_specific_kwargs
    env_cls = _sibling_module("env").VizDoomEnvManager
    # Default to "human-view" parity: the model sees exactly what a human player
    # sees on screen (first-person view + the on-screen HUD), and nothing else.
    # Every oracle channel (depth / labels / objects / sectors / automap) is off.
    # Privileged game variables stay declared for logging/metrics; the
    # observation parser's human_view flag keeps them out of the model's prompt.
    return env_cls(
        config_path="basic.cfg",
        doom_config_path=_vizdoom_runtime_config_path(),
        screen_resolution="RES_320X240",
        screen_format="RGB24",
        available_buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        available_game_variables=["AMMO2", "HEALTH", "ARMOR", "KILLCOUNT", "HITCOUNT", "DAMAGECOUNT", "DAMAGE_TAKEN", "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO"],
        # Human-visible rendering: everything a real player sees on screen.
        render_hud=True,
        render_weapon=True,
        render_messages=True,
        render_screen_flashes=True,
        render_particles=True,
        render_decals=True,
        render_corpses=True,
        render_effects_sprites=True,
        render_crosshair=False,  # vanilla Doom has no crosshair
        # Oracle channels a human never has -> off.
        depth_buffer=False,
        labels_buffer=False,
        automap_buffer=False,
        objects_info=False,
        sectors_info=False,
        notifications_buffer=False,
        audio_buffer=False,
        sound_enabled=False,
        window_visible=False,
        # Show every simulator tic as one five-frame video segment, then ask
        # for exactly one new action for the next five tics.
        frame_history=5,
        tics_per_action=5,
        capture_action_frames=True,
        emit_action_frames=os.getenv("VIZDOOM_EMIT_ACTION_FRAMES", "0").lower() in {"1", "true", "yes", "on"},
        success_reward_min=1.0,
    )


def vizdoom_observation_parser(doc=None, lmms_eval_specific_kwargs=None):
    del doc, lmms_eval_specific_kwargs
    # Human-view parity: the model only gets what a human sees on screen
    # (first-person frame history as video + on-screen HUD), no oracle state.
    return _sibling_module("parsers").VizDoomObservationParser(human_view=True, video=True, image_buffers=["screen"])


def vizdoom_action_parser(doc=None, lmms_eval_specific_kwargs=None):
    del doc, lmms_eval_specific_kwargs
    return _sibling_module("parsers").VizDoomActionParser()


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
    raw = results[0] if results else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"vizdoom_success": 0.0, "vizdoom_steps": 0.0, "vizdoom_invalid_actions": 1.0}

    metrics = payload.get("metrics", {})
    return {
        "vizdoom_success": float(metrics.get("vizdoom_success", 1.0 if payload.get("success") else 0.0)),
        "vizdoom_steps": float(metrics.get("vizdoom_steps", 0.0)),
        "vizdoom_invalid_actions": float(metrics.get("vizdoom_invalid_actions", 0.0)),
    }


def vizdoom_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0

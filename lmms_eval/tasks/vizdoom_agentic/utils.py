import json

from lmms_eval.tasks.vizdoom_agentic.env import VizDoomEnvManager


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
    return VizDoomEnvManager(
        config_path="basic.cfg",
        screen_resolution="RES_320X240",
        screen_format="RGB24",
        available_buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        available_game_variables=["AMMO2", "HEALTH", "KILLCOUNT", "HITCOUNT", "DAMAGECOUNT", "DAMAGE_TAKEN"],
        depth_buffer=True,
        labels_buffer=True,
        automap_buffer=True,
        objects_info=True,
        sectors_info=True,
        notifications_buffer=True,
        notifications_buffer_size=4,
        audio_buffer=False,
        sound_enabled=False,
        window_visible=False,
        frame_history=12,
        tics_per_action=12,
        capture_action_frames=True,
        success_reward_min=1.0,
    )


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


def vizdoom_aggregate_success(results):
    return sum(results) / len(results) if results else 0.0


def vizdoom_aggregate_steps(results):
    return sum(results) / len(results) if results else 0.0


def vizdoom_aggregate_invalid_actions(results):
    return sum(results) / len(results) if results else 0.0

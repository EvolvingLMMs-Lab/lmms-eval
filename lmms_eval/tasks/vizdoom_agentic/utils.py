import json

from lmms_eval.api.agentic import register_game_env
from lmms_eval.tasks.vizdoom_agentic.env import VizDoomEnv


def vizdoom_doc_to_visual(doc):
    return []


def vizdoom_native_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['instruction']}\nUse the VizDoom visual input and state to choose the next action.{post_prompt}"


def vizdoom_native_doc_to_target(doc):
    return "maximize_reward"


register_game_env("vizdoom_native", VizDoomEnv, replace=True)


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

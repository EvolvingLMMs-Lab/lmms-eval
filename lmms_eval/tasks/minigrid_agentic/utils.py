"""MiniGrid task glue: documents in, rollout metrics out."""

import json


def minigrid_doc_to_visual(doc):
    return []


def minigrid_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{doc.get('instruction', '')}{kwargs.get('post_prompt', '')}"


def minigrid_doc_to_target(doc):
    return "reach_goal"


def minigrid_process_results(doc, results):
    raw = results[0] if results else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"minigrid_success": 0.0, "minigrid_steps": 0.0, "minigrid_invalid_actions": 0.0, "minigrid_env_error": 1.0}

    metrics = payload.get("metrics", {})
    return {
        "minigrid_success": 1.0 if payload.get("success") else 0.0,
        "minigrid_steps": float(metrics.get("steps", 0.0)),
        "minigrid_invalid_actions": float(metrics.get("invalid_actions", 0.0)),
        "minigrid_env_error": float(metrics.get("env_error", 0.0)),
    }


def minigrid_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0

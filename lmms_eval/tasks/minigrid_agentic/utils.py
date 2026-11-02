"""MiniGrid as a stateful task on the standard agentic generation loop."""

import json

from lmms_eval.tasks.minigrid_agentic.env import MiniGridEnvManager
from lmms_eval.tasks.minigrid_agentic.parsers import observation_messages, parse_action


def minigrid_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return observation_messages(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)[0]["content"][0]["text"]


def minigrid_doc_to_messages(doc, *, round_idx=None, previous_output=None, previous_round_info=None, lmms_eval_specific_kwargs=None):
    # Request construction and few-shot formatting must never start a simulator.
    if round_idx is None:
        return observation_messages(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)

    outputs = list(previous_output or [])
    info = previous_round_info
    if info is None:
        env = MiniGridEnvManager()
        info = {"env": env, "close": env.close}
        info["finalize"] = lambda: json.dumps(info["state"], ensure_ascii=False)
    env = info["env"]
    try:
        state = env.reset(doc) if round_idx == 0 else env.step(parse_action(outputs[-1] if outputs else ""))
        info["state"] = env.summary()
        messages = [] if state["terminal"] else observation_messages(doc, state, lmms_eval_specific_kwargs)
    except Exception as exc:
        info["state"] = {**env.summary(), "success": False, "terminal": True, "error": str(exc)}
        info["state"]["metrics"] = {**info["state"]["metrics"], "env_error": 1.0}
        if round_idx == 0:
            env.close()
        outputs.append(info["finalize"]())
        return [], True, outputs, info

    if state["terminal"]:
        outputs.append(info["finalize"]())
        return [], True, outputs, info
    return messages, False, outputs, info


def minigrid_doc_to_target(doc):
    return "reach_goal"


def minigrid_process_results(doc, results):
    try:
        payload = json.loads(results[0]) if results else {}
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            raise ValueError("Missing episode summary")
        metrics = payload["metrics"]
        return {
            "minigrid_success": float(bool(payload.get("success"))),
            "minigrid_steps": float(metrics.get("steps", 0.0)),
            "minigrid_invalid_actions": float(metrics.get("invalid_actions", 0.0)),
            "minigrid_env_error": float(metrics.get("env_error", 0.0)),
        }
    except (ValueError, TypeError, KeyError):
        return {"minigrid_success": 0.0, "minigrid_steps": 0.0, "minigrid_invalid_actions": 0.0, "minigrid_env_error": 1.0}


def minigrid_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0

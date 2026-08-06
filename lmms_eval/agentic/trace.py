"""Serialize an episode into the JSON response string handed to the task.

Task ``process_results`` receives ``episode_to_json`` output:

    {"success": bool | null,
     "metrics": {...},
     "final_state": {...},
     "steps": [{"step_idx", "raw_model_output", "model_output", "action",
                "parse_error", "reward", "done", "info"}],
     "metadata": {...}}

``safe_data`` compresses non-JSON payloads (arrays, tensors, PIL images) into
small type/shape descriptors instead of dumping raw pixels into the log.
"""

from __future__ import annotations

import json
from typing import Any

from lmms_eval.agentic.types import EnvState, EpisodeResult, EpisodeStep, GameAction


def episode_to_json(result: EpisodeResult) -> str:
    payload = {
        "success": result.success,
        "metrics": safe_data(result.metrics),
        "final_state": _state_to_dict(result.final_state),
        "steps": [_step_to_dict(step) for step in result.steps],
        "metadata": safe_data(result.metadata),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _step_to_dict(step: EpisodeStep) -> dict[str, Any]:
    return {
        "step_idx": step.state.step_idx,
        "raw_model_output": payload_to_compact_trace(step.raw_output),
        "model_output": payload_to_compact_trace(step.output),
        "action": action_to_dict(step.parsed_action.action) if step.parsed_action is not None else None,
        "parse_error": step.parsed_action.error if step.parsed_action is not None else None,
        "reward": safe_data(step.result.reward) if step.result is not None else None,
        "done": step.result.done if step.result is not None else None,
        "info": safe_data(info_without_frames(step.result.info)) if step.result is not None else {},
    }


def _state_to_dict(state: EnvState) -> dict[str, Any]:
    return {
        "env_id": state.env_id,
        "step_idx": state.step_idx,
        "observation": safe_data(state.observation),
        "terminal": state.terminal,
        "metadata": safe_data(state.metadata),
    }


def action_to_dict(action: Any | None) -> Any | None:
    if action is None:
        return None
    if isinstance(action, GameAction):
        return {"type": action.type, "data": safe_data(action.data), "agent_id": action.agent_id, "metadata": safe_data(action.metadata)}
    if isinstance(action, dict):
        return {_safe_key(agent_id): action_to_dict(agent_action) for agent_id, agent_action in action.items()}
    return safe_data(action)


def action_label(action: Any | None) -> str:
    """Human-readable one-token action summary for logs and artifacts."""

    if action is None:
        return "NONE"
    if isinstance(action, GameAction):
        data = action.data if isinstance(action.data, dict) else {}
        buttons = data.get("buttons")
        if isinstance(buttons, dict):
            active = [name for name, value in buttons.items() if value]
            return "+".join(active) if active else "NOOP"
        if isinstance(buttons, list):
            return "+".join(str(item) for item in buttons) if buttons else "NOOP"
        return action.type
    if isinstance(action, dict):
        return ",".join(f"{agent_id}:{action_label(agent_action)}" for agent_id, agent_action in action.items())
    summary = safe_data(action)
    if isinstance(summary, str):
        return summary
    return _safe_repr(summary, max_chars=120)


def payload_to_compact_trace(value: Any) -> Any:
    if value is None:
        return None
    text = _payload_first_text(value)
    return text if text is not None else safe_data(value)


def info_without_frames(info: Any) -> Any:
    """Drop raw ``action_frames`` arrays before serialization (they go to mp4 artifacts)."""

    if isinstance(info, dict) and "action_frames" in info:
        return {key: value for key, value in info.items() if key != "action_frames"}
    return info


def _payload_first_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    first_text = getattr(value, "first_text", None)
    if not callable(first_text):
        return None
    try:
        text = first_text()
    except Exception:
        return None
    return text if isinstance(text, str) else None


def safe_data(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "length": len(value)}

    scalar = _numpy_scalar_to_python(value)
    if scalar is not value:
        return scalar

    if _looks_like_array_or_tensor(value):
        return {
            "type": _type_name(value),
            "shape": _safe_shape(getattr(value, "shape", None)),
            "dtype": str(getattr(value, "dtype", "")),
        }
    if _looks_like_pil_image(value):
        return {
            "type": _type_name(value),
            "size": safe_data(getattr(value, "size", None), depth=depth + 1, seen=seen),
            "mode": getattr(value, "mode", None),
            "format": getattr(value, "format", None),
        }

    if depth >= 8:
        return {"type": _type_name(value), "repr": _safe_repr(value)}

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return {"type": _type_name(value), "repr": "<recursive>"}
    seen.add(value_id)
    try:
        if isinstance(value, dict):
            return {_safe_key(key): safe_data(item, depth=depth + 1, seen=seen) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe_data(item, depth=depth + 1, seen=seen) for item in value]
        if isinstance(value, set):
            return [safe_data(item, depth=depth + 1, seen=seen) for item in sorted(value, key=repr)]
    finally:
        seen.discard(value_id)

    return {"type": _type_name(value), "repr": _safe_repr(value)}


def _safe_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(safe_data(value))


def _numpy_scalar_to_python(value: Any) -> Any:
    module = getattr(value.__class__, "__module__", "")
    if not module.startswith("numpy"):
        return value
    item = getattr(value, "item", None)
    if not callable(item):
        return value
    try:
        scalar = item()
    except Exception:
        return value
    if scalar is None or isinstance(scalar, str | int | float | bool):
        return scalar
    return value


def _looks_like_array_or_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _looks_like_pil_image(value: Any) -> bool:
    return getattr(value.__class__, "__module__", "").startswith("PIL.") and hasattr(value, "size") and hasattr(value, "mode")


def _safe_shape(shape: Any) -> Any:
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except TypeError:
        return str(shape)


def _type_name(value: Any) -> str:
    cls = value.__class__
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__qualname__", cls.__name__)
    return name if module in {"", "builtins"} else f"{module}.{name}"


def _safe_repr(value: Any, max_chars: int = 500) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {_type_name(value)}>"
    if len(text) > max_chars:
        return f"{text[:max_chars]}...<truncated>"
    return text

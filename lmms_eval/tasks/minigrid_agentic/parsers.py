"""Task-local ``Any -> Any`` parser functions for MiniGrid rollouts."""

from __future__ import annotations

import json
import re
from typing import Any

from lmms_eval.agentic import (
    ActionSpec,
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    GameAction,
    ParsedAction,
    ParserContext,
)


def minigrid_observation_parser(value: Any, context: ParserContext) -> Any:
    """Render MiniGrid state, action help, and frames into one model request."""

    if not isinstance(value, EnvState):
        raise TypeError(f"minigrid_observation_parser requires EnvState, got {type(value).__name__}")
    observation = value.observation if isinstance(value.observation, dict) else {"text": "" if value.observation is None else str(value.observation)}
    spec = context.metadata.get("action_spec")
    doc = context.metadata.get("doc")
    max_steps = context.metadata.get("max_steps")
    variables = observation.get("variables")
    actions = observation.get("actions")
    if not isinstance(actions, str):
        actions = spec.render_prompt() if isinstance(spec, ActionSpec) else ""

    sections = [
        str(doc.get("instruction") or "") if isinstance(doc, dict) else "",
        str(observation.get("text") or ""),
        f"Variables: {json.dumps(variables, ensure_ascii=False, sort_keys=True, default=str)}" if isinstance(variables, dict) and variables else "",
        f"Step {value.step_idx} of {max_steps}." if max_steps is not None else f"Step {value.step_idx}.",
        f"Available actions:\n{actions}" if actions else "",
        _action_directive(spec) if actions else "",
    ]
    content = [ContentBlock.text("\n\n".join(section for section in sections if section).strip())]
    frames = observation.get("video")
    if _has_frames(frames):
        content.append(ContentBlock(type="video", data=list(frames), metadata={"source": "video"}))
    images = observation.get("images")
    if _has_frames(images):
        content.extend(ContentBlock(type="image", data=image, metadata={"source": "images"}) for image in images)
    return AgentInput(content=content, metadata={"env_id": value.env_id, "step_idx": value.step_idx, "agent_id": context.agent_id})


def minigrid_qwen_output_parser(value: Any, context: ParserContext) -> Any:
    """Remove a completed Qwen thinking block before task action parsing."""

    del context
    if not isinstance(value, AgentOutput):
        raise TypeError(f"minigrid_qwen_output_parser requires AgentOutput, got {type(value).__name__}")
    text = value.first_text() or ""
    normalized = _strip_thinking(text)
    metadata = {**value.metadata, "raw_text": text, "normalized_text": normalized}
    content = []
    replaced_text = False
    for block in value.content:
        if not replaced_text and block.type == "text" and isinstance(block.data, str):
            content.append(ContentBlock.text(normalized, **{**block.metadata, "raw_text": block.data}))
            replaced_text = True
        else:
            content.append(block)
    if not replaced_text:
        content.append(ContentBlock.text(normalized))
    return AgentOutput(content=content, metadata=metadata)


def minigrid_action_parser(value: Any, context: ParserContext) -> Any:
    """Parse a declared MiniGrid action name from model output."""

    if not isinstance(value, AgentOutput):
        return ParsedAction(error=f"minigrid_action_parser requires AgentOutput, got {type(value).__name__}")
    spec = context.metadata.get("action_spec")
    if not isinstance(spec, ActionSpec):
        return ParsedAction(error="MiniGrid environment did not provide an ActionSpec")
    text = value.first_text() or ""
    action_name = _extract_action_name(text, value.metadata, spec)
    if action_name is None:
        return ParsedAction(error="no valid MiniGrid action found", metadata={"raw_output": text})
    return ParsedAction(
        action=GameAction(type=action_name, agent_id=context.agent_id, metadata={"raw_output": text}),
        is_submit=action_name in {name.upper() for name in spec.submit_actions},
    )


def _extract_action_name(text: str, metadata: dict[str, Any], spec: ActionSpec) -> str | None:
    valid = {name.upper() for name in spec.action_names()} | {name.upper() for name in spec.submit_actions}
    aliases = {alias.upper(): name.upper() for alias, name in spec.alias_map().items()}
    tool_calls = metadata.get("tool_calls")
    if isinstance(tool_calls, list):
        for mapping in tool_calls:
            if isinstance(mapping, dict):
                action = _action_from_mapping(mapping, valid, aliases)
                if action is not None:
                    return action
    for mapping in _extract_json_objects(text):
        action = _action_from_mapping(mapping, valid, aliases)
        if action is not None:
            return action
    upper_text = text.upper()
    for candidate in sorted(valid | set(aliases), key=len, reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", upper_text):
            return aliases.get(candidate, candidate)
    return None


def _action_from_mapping(mapping: dict[str, Any], valid: set[str], aliases: dict[str, str]) -> str | None:
    candidates = [mapping.get(key) for key in ("action", "action_name", "name", "tool_name", "type")]
    arguments = mapping.get("arguments")
    if isinstance(arguments, dict):
        candidates.extend(arguments.get(key) for key in ("action", "action_name", "name", "type"))
    for candidate in candidates:
        if isinstance(candidate, str):
            normalized = aliases.get(candidate.strip().upper(), candidate.strip().upper())
            if normalized in valid:
                return normalized
    return None


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def _action_directive(spec: Any) -> str:
    if not isinstance(spec, ActionSpec):
        return ""
    return {
        "discrete": "Respond with only the action name.",
        "parameterized": 'Respond with a single JSON object: {"action": <name>, ...arguments}.',
        "free_text": "Respond with a single short text command.",
    }.get(spec.kind, "")


def _strip_thinking(text: str) -> str:
    candidate = text.strip()
    if "</think>" in candidate:
        after_thinking = candidate.rsplit("</think>", 1)[-1].strip()
        if after_thinking:
            return after_thinking
        return re.sub(r"</?think>", "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _has_frames(value: Any) -> bool:
    if value is None:
        return False
    length = getattr(value, "__len__", None)
    return bool(len(value)) if callable(length) else True

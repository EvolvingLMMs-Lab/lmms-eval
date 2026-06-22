from __future__ import annotations

import json
import re
from typing import Any

from lmms_eval.agentic.parsers.base import ActionParser, ParserContext
from lmms_eval.agentic.registry_core import register_action_parser
from lmms_eval.agentic.types import AgentOutput, GameAction, ParsedAction
from lmms_eval.imports import optional_import

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ACTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", re.DOTALL)


@register_action_parser("vizdoom", replace=True)
class VizDoomActionParser(ActionParser):
    """Parse skill/tool-call/text output into VizDoom button actions."""

    def __init__(
        self,
        buttons: list[str] | str | None = None,
        submit_actions: list[str] | str | None = ("SUBMIT",),
        noop_actions: list[str] | str | None = ("NOOP",),
        button_skill_names: list[str] | str | None = ("press_buttons", "press_button", "vizdoom_press", "vizdoom_action", "act"),
        noop_skill_names: list[str] | str | None = ("noop",),
        submit_skill_names: list[str] | str | None = ("submit",),
    ) -> None:
        self.buttons = set(_as_list(buttons)) if buttons is not None else _all_vizdoom_buttons()
        self.submit_actions = set(_as_list(submit_actions))
        self.noop_actions = set(_as_list(noop_actions))
        self.button_skill_names = set(_as_list(button_skill_names))
        self.noop_skill_names = set(_as_list(noop_skill_names))
        self.submit_skill_names = set(_as_list(submit_skill_names))

    def parse(self, value: Any, ctx: ParserContext) -> ParsedAction:
        if not isinstance(value, AgentOutput):
            return ParsedAction(error=f"VizDoomActionParser requires AgentOutput, got {type(value).__name__}")
        output = value
        agent_id = ctx.agent_id
        text = output.first_text() or ""
        for tool_call in output.metadata.get("tool_calls", []) if isinstance(output.metadata.get("tool_calls"), list) else []:
            parsed = self._parse_skill_call(tool_call, text, agent_id)
            if parsed is not None:
                return parsed

        for tool_call in _extract_tool_calls(text):
            parsed = self._parse_skill_call(tool_call, text, agent_id)
            if parsed is not None:
                return parsed

        for function_call in _extract_function_calls(text):
            parsed = self._parse_skill_call(function_call, text, agent_id)
            if parsed is not None:
                return parsed

        payload = _extract_json_payload(text)
        if isinstance(payload, dict):
            parsed = self._parse_json_payload(payload, text, agent_id)
            if parsed is not None:
                return parsed

        tokens = [token.upper() for token in _ACTION_TOKEN_RE.findall(text)]
        submit = next((token for token in tokens if token in self.submit_actions), None)
        if submit:
            return ParsedAction(action=GameAction(type=submit, agent_id=agent_id), is_submit=True, metadata={"raw_output": text})

        noop = next((token for token in tokens if token in self.noop_actions), None)
        if noop:
            return ParsedAction(action=GameAction(type="NOOP", agent_id=agent_id), metadata={"raw_output": text})

        buttons = [token for token in tokens if token in self.buttons]
        if not buttons:
            buttons = self._buttons_from_natural_language(text)
        if not buttons:
            return ParsedAction(error="no valid VizDoom button found", metadata={"raw_output": text})
        if len(buttons) == 1:
            return ParsedAction(action=GameAction(type=buttons[0], agent_id=agent_id), metadata={"raw_output": text})
        return ParsedAction(
            action=GameAction(type="vizdoom_action", data={"buttons": buttons}, agent_id=agent_id),
            metadata={"raw_output": text, "buttons": buttons},
        )

    def _parse_skill_call(self, call: dict[str, Any], raw_text: str, agent_id: str | None) -> ParsedAction | None:
        name = _normalize_name(call.get("name") or call.get("tool_name") or call.get("function") or call.get("type"))
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}

        if name in self.submit_skill_names or _argument_action(arguments) in self.submit_actions:
            return ParsedAction(action=GameAction(type="SUBMIT", agent_id=agent_id), is_submit=True, metadata={"raw_output": raw_text, "skill_call": call})
        if name in self.noop_skill_names or _argument_action(arguments) in self.noop_actions:
            return ParsedAction(action=GameAction(type="NOOP", agent_id=agent_id, metadata=_tics_metadata(arguments)), metadata={"raw_output": raw_text, "skill_call": call})
        if name in self.buttons:
            return ParsedAction(action=GameAction(type=name, agent_id=agent_id, metadata=_tics_metadata(arguments)), metadata={"raw_output": raw_text, "skill_call": call})

        if name and name not in self.button_skill_names:
            return None

        buttons = self._buttons_from_arguments(arguments)
        if buttons is None:
            return None
        if isinstance(buttons, str):
            if buttons in self.submit_actions:
                return ParsedAction(action=GameAction(type=buttons, agent_id=agent_id), is_submit=True, metadata={"raw_output": raw_text, "skill_call": call})
            if buttons in self.noop_actions:
                return ParsedAction(action=GameAction(type="NOOP", agent_id=agent_id, metadata=_tics_metadata(arguments)), metadata={"raw_output": raw_text, "skill_call": call})
            if buttons in self.buttons:
                return ParsedAction(action=GameAction(type=buttons, agent_id=agent_id, metadata=_tics_metadata(arguments)), metadata={"raw_output": raw_text, "skill_call": call})
            return None

        data = {"buttons": buttons, **_tics_data(arguments)}
        return ParsedAction(action=GameAction(type="vizdoom_action", data=data, agent_id=agent_id), metadata={"raw_output": raw_text, "skill_call": call, "buttons": buttons})

    def _parse_json_payload(self, payload: dict[str, Any], raw_text: str, agent_id: str | None) -> ParsedAction | None:
        if payload.get("submit") is True:
            return ParsedAction(action=GameAction(type="SUBMIT", agent_id=agent_id), is_submit=True, metadata={"raw_output": raw_text, "json": payload})

        action = payload.get("action")
        if isinstance(action, str):
            action_name = action.upper()
            if action_name in self.submit_actions:
                return ParsedAction(action=GameAction(type=action_name, agent_id=agent_id), is_submit=True, metadata={"raw_output": raw_text, "json": payload})
            if action_name in self.noop_actions:
                return ParsedAction(action=GameAction(type="NOOP", agent_id=agent_id), metadata={"raw_output": raw_text, "json": payload})
            if action_name in self.buttons:
                return ParsedAction(action=GameAction(type=action_name, data=_numeric_or_none(payload.get("value")), agent_id=agent_id, metadata=_tics_metadata(payload)), metadata={"raw_output": raw_text, "json": payload})

        buttons = payload.get("buttons", payload.get("button_values", payload.get("actions")))
        if isinstance(buttons, str):
            buttons = [buttons]
        if isinstance(buttons, list):
            normalized = [str(button).upper() for button in buttons]
            valid = [button for button in normalized if button in self.buttons]
            if valid:
                return ParsedAction(action=GameAction(type="vizdoom_action", data={"buttons": valid, **_tics_data(payload)}, agent_id=agent_id), metadata={"raw_output": raw_text, "json": payload, "buttons": valid})
        if isinstance(buttons, dict):
            normalized = {str(button).upper(): value for button, value in buttons.items() if str(button).upper() in self.buttons}
            if normalized:
                return ParsedAction(action=GameAction(type="vizdoom_action", data={"buttons": normalized, **_tics_data(payload)}, agent_id=agent_id), metadata={"raw_output": raw_text, "json": payload, "buttons": normalized})

        values = payload.get("values")
        if isinstance(values, list):
            return ParsedAction(action=GameAction(type="button_vector", data={"values": values, **_tics_data(payload)}, agent_id=agent_id), metadata={"raw_output": raw_text, "json": payload})
        return None

    def _buttons_from_arguments(self, arguments: dict[str, Any]) -> str | list[str] | dict[str, Any] | None:
        candidates = [
            arguments.get("buttons"),
            arguments.get("button"),
            arguments.get("button_values"),
            arguments.get("actions"),
            arguments.get("action"),
            arguments.get("name"),
            arguments.get("_args"),
        ]
        for candidate in candidates:
            parsed = self._normalize_buttons(candidate)
            if parsed is not None:
                return parsed
        return None

    def _normalize_buttons(self, value: Any) -> str | list[str] | dict[str, Any] | None:
        if isinstance(value, dict):
            normalized = {str(button).upper(): amount for button, amount in value.items() if str(button).upper() in self.buttons}
            return normalized or None
        if isinstance(value, list | tuple):
            normalized = [str(button).upper() for button in value if str(button).upper() in self.buttons]
            return normalized or None
        if isinstance(value, str):
            action = value.strip().upper()
            if action in self.submit_actions or action in self.noop_actions or action in self.buttons:
                return action
            normalized = [token.upper() for token in _ACTION_TOKEN_RE.findall(value) if token.upper() in self.buttons]
            return normalized or None
        return None

    def _buttons_from_natural_language(self, text: str) -> list[str]:
        normalized = text.upper()
        buttons = []
        if "ATTACK" in self.buttons and re.search(r"\b(SHOOT|SHOOTING|SHOT|SHOTS|FIRE|FIRING)\b", normalized):
            buttons.append("ATTACK")
        if "MOVE_LEFT" in self.buttons and re.search(r"\b(MOVE|STRAFE|GO|TURN)\s+LEFT\b", normalized):
            buttons.append("MOVE_LEFT")
        if "MOVE_RIGHT" in self.buttons and re.search(r"\b(MOVE|STRAFE|GO|TURN)\s+RIGHT\b", normalized):
            buttons.append("MOVE_RIGHT")
        return buttons

def _extract_json_payload(text: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    tool_calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        payload = match.group(1)
        function_match = re.search(r"<function=([^>\s]+)>(.*?)</function>", payload, flags=re.DOTALL | re.IGNORECASE)
        if function_match is None:
            continue
        params = {}
        for param_match in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", function_match.group(2), flags=re.DOTALL | re.IGNORECASE):
            params[param_match.group(1)] = param_match.group(2).strip()
        tool_calls.append({"name": function_match.group(1), "arguments": params})
    return tool_calls


def _extract_function_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in _FUNCTION_CALL_RE.finditer(text):
        name = match.group(1)
        args_text = match.group(2).strip()
        arguments = {"_args": args_text}
        for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\)]+)", args_text):
            arguments[key] = value.strip().strip("\"'")
        calls.append({"name": name, "arguments": arguments})
    return calls


def _all_vizdoom_buttons() -> set[str]:
    vizdoom, has_vizdoom = optional_import("vizdoom")
    if not has_vizdoom:
        return set()
    return set(getattr(vizdoom.Button, "__members__", {}).keys())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    return [str(item).upper() for item in value]


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper()


def _argument_action(arguments: dict[str, Any]) -> str | None:
    for key in ("action", "name", "type"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value.strip().upper()
    return None


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tics_data(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"tics": int(payload["tics"])} if "tics" in payload else {}
    except (TypeError, ValueError):
        return {}


def _tics_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return _tics_data(payload)

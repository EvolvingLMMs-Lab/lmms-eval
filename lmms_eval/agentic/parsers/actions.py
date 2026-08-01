"""Generic action parsers built from an environment's ``ActionSpec``."""

from __future__ import annotations

import json
import re
from typing import Any

from lmms_eval.agentic.parsers.base import ActionParser, ParserContext
from lmms_eval.agentic.types import (
    ActionDef,
    ActionSpec,
    AgentOutput,
    GameAction,
    ParsedAction,
)


class ActionNameParser(ActionParser):
    """Parse one action name from text, JSON, or tool-call-like model outputs."""

    def __init__(
        self,
        actions: list[str] | tuple[str, ...] | set[str],
        submit_actions: list[str] | tuple[str, ...] | set[str] = ("SUBMIT",),
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.actions = {action.upper() for action in actions}
        self.submit_actions = {action.upper() for action in submit_actions}
        self.aliases = {key.upper(): value.upper() for key, value in (aliases or {}).items()}
        self.valid_actions = self.actions | self.submit_actions

    @classmethod
    def from_spec(cls, spec: ActionSpec) -> "ActionNameParser":
        return cls(actions=spec.action_names(), submit_actions=tuple(spec.submit_actions), aliases=spec.alias_map())

    def parse(self, output: AgentOutput, ctx: ParserContext) -> ParsedAction:
        if not isinstance(output, AgentOutput):
            return ParsedAction(error=f"ActionNameParser requires AgentOutput, got {type(output).__name__}")
        text = output.first_text() or ""
        action_name = self._extract_action_name(text, output.metadata)
        if action_name is None:
            return ParsedAction(error="no valid action found", metadata={"raw_output": text})
        return ParsedAction(
            action=GameAction(type=action_name, agent_id=ctx.agent_id, metadata={"raw_output": text}),
            is_submit=action_name in self.submit_actions,
        )

    def _extract_action_name(self, text: str, metadata: dict[str, Any]) -> str | None:
        for tool_call in metadata.get("tool_calls", []) if isinstance(metadata.get("tool_calls"), list) else []:
            action_name = self._action_from_mapping(tool_call)
            if action_name is not None:
                return action_name

        for json_candidate in _extract_json_objects(text):
            action_name = self._action_from_mapping(json_candidate)
            if action_name is not None:
                return action_name

        for xml_action in re.findall(r"<parameter=action>\s*(.*?)\s*</parameter>", text, flags=re.DOTALL | re.IGNORECASE):
            action_name = self._normalize_action(xml_action)
            if action_name is not None:
                return action_name

        upper_text = text.upper()
        for candidate in sorted(self.valid_actions | set(self.aliases), key=len, reverse=True):
            if re.search(rf"\b{re.escape(candidate)}\b", upper_text):
                return self.aliases.get(candidate, candidate)
        return None

    def _action_from_mapping(self, mapping: dict[str, Any]) -> str | None:
        candidates = [
            mapping.get("action"),
            mapping.get("action_name"),
            mapping.get("name"),
            mapping.get("tool_name"),
            mapping.get("type"),
        ]
        arguments = mapping.get("arguments")
        if isinstance(arguments, dict):
            candidates.extend([arguments.get("action"), arguments.get("action_name"), arguments.get("name"), arguments.get("type")])

        for candidate in candidates:
            if isinstance(candidate, str):
                action_name = self._normalize_action(candidate)
                if action_name is not None:
                    return action_name
        return None

    def _normalize_action(self, candidate: str) -> str | None:
        action_name = candidate.strip().upper()
        action_name = self.aliases.get(action_name, action_name)
        return action_name if action_name in self.valid_actions else None


class SchemaActionParser(ActionParser):
    """Parse ``{"action": <name>, ...arguments}`` outputs against an ``ActionSpec``.

    Accepts tool calls (``metadata["tool_calls"]``) and inline JSON objects.
    Arguments are checked against the matched ``ActionDef.schema`` (required
    keys and primitive types only; this is deliberately not a full JSON-Schema
    validator). Zero-argument actions also match as bare names in text.
    """

    _NAME_KEYS = ("action", "action_name", "name", "tool_name", "type")

    def __init__(self, spec: ActionSpec | None = None, actions: list[Any] | None = None, submit_actions: list[str] | tuple[str, ...] = (), prompt_hint: str | None = None) -> None:
        if spec is None:
            spec = ActionSpec(kind="parameterized", actions=[_as_action_def(action) for action in actions or []], submit_actions=list(submit_actions), prompt_hint=prompt_hint)
        self.spec = spec
        self.submit_actions = {name.upper() for name in spec.submit_actions}
        self._by_name = {action.name.upper(): action for action in spec.actions}
        self._by_name.update({alias.upper(): self._by_name[target.upper()] for alias, target in spec.alias_map().items() if target.upper() in self._by_name})

    def parse(self, output: AgentOutput, ctx: ParserContext) -> ParsedAction:
        if not isinstance(output, AgentOutput):
            return ParsedAction(error=f"SchemaActionParser requires AgentOutput, got {type(output).__name__}")
        text = output.first_text() or ""
        errors: list[str] = []

        candidates = [*_metadata_tool_calls(output.metadata), *_extract_json_objects(text)]
        for mapping in candidates:
            action_def, arguments = self._match_mapping(mapping)
            if action_def is None:
                continue
            error = _check_arguments(action_def, arguments)
            if error is not None:
                errors.append(f"{action_def.name}: {error}")
                continue
            return self._parsed(action_def, arguments, ctx, raw_output=text)

        bare = self._match_bare_name(text)
        if bare is not None:
            return self._parsed(bare, {}, ctx, raw_output=text)

        detail = f" ({'; '.join(errors)})" if errors else ""
        return ParsedAction(error=f"no valid action found{detail}", metadata={"raw_output": text})

    def _parsed(self, action_def: ActionDef, arguments: dict[str, Any], ctx: ParserContext, *, raw_output: str) -> ParsedAction:
        action = GameAction(type=action_def.name, data=arguments or None, agent_id=ctx.agent_id, metadata={"raw_output": raw_output})
        return ParsedAction(action=action, is_submit=action_def.name.upper() in self.submit_actions)

    def _match_mapping(self, mapping: dict[str, Any]) -> tuple[ActionDef | None, dict[str, Any]]:
        if not isinstance(mapping, dict):
            return None, {}
        for key in self._NAME_KEYS:
            name = mapping.get(key)
            if not isinstance(name, str):
                continue
            action_def = self._by_name.get(name.strip().upper())
            if action_def is None:
                continue
            arguments = mapping.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {k: v for k, v in mapping.items() if k not in self._NAME_KEYS and k not in {"arguments", "id"}}
            return action_def, dict(arguments)
        return None, {}

    def _match_bare_name(self, text: str) -> ActionDef | None:
        upper_text = text.upper()
        for key in sorted(self._by_name, key=len, reverse=True):
            action_def = self._by_name[key]
            if (action_def.schema or {}).get("required"):
                continue
            if re.search(rf"\b{re.escape(key)}\b", upper_text):
                return action_def
        return None


class FreeTextActionParser(ActionParser):
    """Whole model reply becomes one text command; the environment validates it."""

    def __init__(self, spec: ActionSpec | None = None, submit_actions: list[str] | tuple[str, ...] = (), action_type: str = "text_command") -> None:
        submit = list(spec.submit_actions) if spec is not None else list(submit_actions)
        self.submit_actions = {name.upper() for name in submit}
        self.action_type = action_type

    def parse(self, output: AgentOutput, ctx: ParserContext) -> ParsedAction:
        if not isinstance(output, AgentOutput):
            return ParsedAction(error=f"FreeTextActionParser requires AgentOutput, got {type(output).__name__}")
        text = (output.first_text() or "").strip()
        if not text:
            return ParsedAction(error="empty output")
        return ParsedAction(
            action=GameAction(type=self.action_type, data=text, agent_id=ctx.agent_id),
            is_submit=text.upper() in self.submit_actions,
        )


def build_action_parser(spec: ActionSpec | None) -> ActionParser:
    """Default action parser for tasks whose YAML omits ``action_parser``."""

    if spec is None:
        raise ValueError("Task YAML omits action_parser and the environment declares no action_spec(); add action_parser: !function ... to the task or implement EnvManager.action_spec().")
    if spec.kind == "discrete":
        return ActionNameParser.from_spec(spec)
    if spec.kind == "parameterized":
        return SchemaActionParser(spec)
    if spec.kind == "free_text":
        return FreeTextActionParser(spec)
    raise ValueError(f"Unknown ActionSpec.kind {spec.kind!r}; expected discrete, parameterized, or free_text.")


def _as_action_def(value: Any) -> ActionDef:
    if isinstance(value, ActionDef):
        return value
    if isinstance(value, dict):
        return ActionDef(name=str(value["name"]), description=value.get("description"), schema=value.get("schema"), aliases=list(value.get("aliases") or []))
    return ActionDef(name=str(value))


def _check_arguments(action_def: ActionDef, arguments: dict[str, Any]) -> str | None:
    schema = action_def.schema or {}
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in arguments:
            return f"missing required argument {name!r}"
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type") if isinstance(properties.get(name), dict) else None
        if expected is not None and not _matches_json_type(value, expected):
            return f"argument {name!r} should be {expected}, got {type(value).__name__}"
    return None


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _matches_json_type(value: Any, expected: str) -> bool:
    python_type = _JSON_TYPES.get(expected)
    if python_type is None:
        return True
    if expected in {"number", "integer"} and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def _metadata_tool_calls(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = metadata.get("tool_calls")
    return list(tool_calls) if isinstance(tool_calls, list) else []


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

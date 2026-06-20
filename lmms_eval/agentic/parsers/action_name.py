from __future__ import annotations

import json
import re
from typing import Any

from lmms_eval.agentic.parsers.base import ActionParser
from lmms_eval.agentic.types import AgentOutput, EnvState, GameAction, ParsedAction


class ActionNameParser(ActionParser):
    """Parse one action from text, JSON, or tool-call-like model outputs."""

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

    def parse(self, output: AgentOutput, state: EnvState, agent_id: str | None = None) -> ParsedAction:
        del state
        text = output.first_text() or ""
        action_name = self._extract_action_name(text, output.metadata)
        if action_name is None:
            return ParsedAction(error="no valid action found", metadata={"raw_output": text})
        return ParsedAction(
            action=GameAction(type=action_name, agent_id=agent_id, metadata={"raw_output": text}),
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
        for action_name in sorted(self.valid_actions, key=len, reverse=True):
            if re.search(rf"\b{re.escape(action_name)}\b", upper_text):
                return action_name
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
            candidates.extend(
                [
                    arguments.get("action"),
                    arguments.get("action_name"),
                    arguments.get("name"),
                    arguments.get("type"),
                ]
            )

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

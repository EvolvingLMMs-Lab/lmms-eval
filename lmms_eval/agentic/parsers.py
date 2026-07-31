"""Parser stages of the agentic game loop.

Three typed roles connect environment and model:

- ``ObservationParser``:  ``EnvState``  -> ``AgentInput``  (task-side, YAML)
- ``ModelOutputParser``:  ``AgentOutput`` -> ``AgentOutput`` (model-side, CLI)
- ``ActionParser``:       ``AgentOutput`` -> ``ParsedAction`` (task-side, YAML)

Non-text payloads (tensors, latents) travel inside ``ContentBlock``s, not by
loosening these signatures. Task-specific parsers live next to their task;
this module only ships the generic ones.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    GameAction,
    ParsedAction,
)


@dataclass(slots=True)
class ParserContext:
    """Side-channel rollout state passed to every parser call."""

    state: EnvState | None = None
    agent_id: str | None = None
    step_idx: int | None = None
    request: AgentInput | None = None
    raw_output: AgentOutput | None = None
    history: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservationParser(ABC):
    """Environment state -> model request."""

    @abstractmethod
    def parse(self, state: EnvState, ctx: ParserContext) -> AgentInput:
        raise NotImplementedError


class ModelOutputParser(ABC):
    """Raw model output -> normalized model output."""

    @abstractmethod
    def parse(self, output: AgentOutput, ctx: ParserContext) -> AgentOutput:
        raise NotImplementedError


class ActionParser(ABC):
    """Normalized model output -> environment action."""

    @abstractmethod
    def parse(self, output: AgentOutput, ctx: ParserContext) -> ParsedAction:
        raise NotImplementedError


class IdentityModelOutputParser(ModelOutputParser):
    """Pass model output through unchanged."""

    def parse(self, output: AgentOutput, ctx: ParserContext) -> AgentOutput:
        del ctx
        return output


class QwenModelOutputParser(ModelOutputParser):
    """Normalize common Qwen chat outputs before task action parsing."""

    def __init__(self, strip_thinking: bool = True, extract_tool_calls: bool = True) -> None:
        self.strip_thinking = strip_thinking
        self.extract_tool_calls = extract_tool_calls

    def parse(self, output: AgentOutput, ctx: ParserContext) -> AgentOutput:
        del ctx
        if not isinstance(output, AgentOutput):
            raise TypeError(f"QwenModelOutputParser requires AgentOutput, got {type(output).__name__}")
        text = output.first_text() or ""
        normalized_text = _strip_thinking(text) if self.strip_thinking else text
        metadata = dict(output.metadata)
        metadata["raw_text"] = text
        metadata["normalized_text"] = normalized_text
        if self.extract_tool_calls:
            metadata["tool_calls"] = [*_metadata_tool_calls(metadata), *_extract_qwen_tool_calls(text)]

        content = []
        replaced_text = False
        for block in output.content:
            if not replaced_text and block.type == "text" and isinstance(block.data, str):
                block_metadata = dict(block.metadata)
                block_metadata["raw_text"] = block.data
                content.append(ContentBlock.text(normalized_text, **block_metadata))
                replaced_text = True
            else:
                content.append(block)
        if not replaced_text:
            content.append(ContentBlock.text(normalized_text))
        return AgentOutput(content=content, metadata=metadata)


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


def _strip_thinking(text: str) -> str:
    candidate = text.strip()
    if "</think>" in candidate:
        after_thinking = candidate.rsplit("</think>", 1)[-1].strip()
        if after_thinking:
            return after_thinking
        # Some thinking-mode responses stop right after the closing marker.
        # Keep the reasoning text so action parsers can still recover a name.
        return re.sub(r"</?think>", "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _extract_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
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

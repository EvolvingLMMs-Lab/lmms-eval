"""Parser stages of the agentic game loop.

Three typed roles connect environment and model:

- ``ObservationParser``:  ``EnvState``  -> ``AgentInput``  (task-side, YAML)
- ``ModelOutputParser``:  ``AgentOutput`` -> ``AgentOutput`` (model-side, CLI)
- ``ActionParser``:       ``AgentOutput`` -> ``ParsedAction`` (task-side, YAML)

Non-text payloads (tensors, latents) travel inside ``ContentBlock``s, not by
loosening these signatures. Task-specific parsers live next to their task;
this module only ships the generic ones.

A task YAML may omit both task-side parsers: the loop then falls back to
``TemplateObservationParser`` (which reads the reserved observation keys
``text`` / ``images`` / ``video`` / ``variables`` / ``actions``) and to
``build_action_parser(env.action_spec())``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lmms_eval.agentic.types import (
    ActionDef,
    ActionSpec,
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


class TemplateObservationParser(ObservationParser):
    """Render reserved observation keys into an ``AgentInput`` with no task code.

    Reserved keys in a dict ``EnvState.observation``: ``text`` (str),
    ``images`` (list of frames), ``video`` (list of frames for one clip),
    ``variables`` (dict), ``actions`` (pre-rendered action text overriding the
    env's ``action_spec()``). Placeholders available in ``template``:
    ``{instruction}`` ``{text}`` ``{variables}`` ``{actions}`` ``{directive}``
    ``{step_idx}`` ``{max_steps}``. Without ``template``, empty sections are
    dropped instead of rendering blank lines.
    """

    _DEFAULT_SECTIONS = ("{instruction}", "{text}", "Variables: {variables}", "{step_line}", "Available actions:\n{actions}", "{directive}")
    _DIRECTIVES = {
        "discrete": "Respond with only the action name.",
        "parameterized": 'Respond with a single JSON object: {"action": <name>, ...arguments}.',
        "free_text": "Respond with a single short text command.",
    }

    def __init__(self, template: str | None = None, include_images: bool | str = True, include_video: bool | str = True, max_images: int | None = None) -> None:
        self.template = template
        self.include_images = _as_flag(include_images)
        self.include_video = _as_flag(include_video)
        self.max_images = int(max_images) if max_images is not None else None

    def parse(self, state: EnvState, ctx: ParserContext) -> AgentInput:
        if not isinstance(state, EnvState):
            raise TypeError(f"TemplateObservationParser requires EnvState, got {type(state).__name__}")
        observation = state.observation if isinstance(state.observation, dict) else {"text": "" if state.observation is None else str(state.observation)}
        fields = self._fields(observation, state, ctx)

        content = [ContentBlock.text(self._render(fields))]
        if self.include_video:
            frames = observation.get("video")
            if _has_frames(frames):
                content.append(ContentBlock(type="video", data=list(frames), metadata={"source": "video"}))
        if self.include_images:
            images = observation.get("images")
            if _has_frames(images):
                images = list(images)
                if self.max_images is not None:
                    images = images[-self.max_images :]
                content.extend(ContentBlock(type="image", data=image, metadata={"source": "images"}) for image in images)

        return AgentInput(content=content, metadata={"env_id": state.env_id, "step_idx": state.step_idx, "agent_id": ctx.agent_id})

    def _fields(self, observation: dict[str, Any], state: EnvState, ctx: ParserContext) -> dict[str, str]:
        doc = ctx.metadata.get("doc")
        spec = ctx.metadata.get("action_spec")
        variables = observation.get("variables")
        actions = observation.get("actions")
        if not isinstance(actions, str):
            actions = spec.render_prompt() if isinstance(spec, ActionSpec) else ""
        max_steps = ctx.metadata.get("max_steps")
        step_line = f"Step {state.step_idx} of {max_steps}." if max_steps is not None else f"Step {state.step_idx}."
        return {
            "instruction": str(doc.get("instruction") or "") if isinstance(doc, dict) else "",
            "text": str(observation.get("text") or ""),
            "variables": json.dumps(variables, ensure_ascii=False, sort_keys=True, default=str) if isinstance(variables, dict) and variables else "",
            "actions": actions,
            "directive": self._DIRECTIVES.get(spec.kind, "") if isinstance(spec, ActionSpec) and actions else "",
            "step_idx": str(state.step_idx),
            "max_steps": "" if max_steps is None else str(max_steps),
            "step_line": step_line,
        }

    def _render(self, fields: dict[str, str]) -> str:
        if self.template is not None:
            rendered = self.template.format_map(_DefaultEmpty(fields))
            return re.sub(r"\n{3,}", "\n\n", rendered).strip()
        sections = []
        for section in self._DEFAULT_SECTIONS:
            placeholders = re.findall(r"\{(\w+)\}", section)
            if all(not fields.get(name) for name in placeholders):
                continue
            sections.append(section.format_map(_DefaultEmpty(fields)))
        return "\n\n".join(sections).strip()


class _DefaultEmpty(dict):
    def __missing__(self, key: str) -> str:
        return ""


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


def _has_frames(value: Any) -> bool:
    if value is None:
        return False
    length = getattr(value, "__len__", None)
    return bool(len(value)) if callable(length) else True


def _as_flag(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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

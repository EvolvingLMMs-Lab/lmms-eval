"""Task-local ``Any -> Any`` parser functions for ViZDoom rollouts."""

from __future__ import annotations

import json
import re
from typing import Any

from lmms_eval.agentic import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    GameAction,
    ParsedAction,
    ParserContext,
)
from lmms_eval.imports import optional_import

_VIZDOOM_BUFFER_KEYS = {
    "screen_buffer",
    "depth_buffer",
    "labels_buffer",
    "automap_buffer",
    "audio_buffer",
    "notifications_buffer",
    "screen_history",
}


class _VizDoomObservationFormatter:
    """Implementation helper behind :func:`vizdoom_observation_parser`."""

    def __init__(
        self,
        image_buffers: list[str] | str | None = ("screen",),
        video: bool = False,
        video_buffer: str = "screen_history",
        include_structured_state: bool = True,
        include_raw_buffers: bool = True,
        human_view: bool | str = False,
        prompt: str | None = None,
        action_format: str = "skill",
        skill_name: str = "press_buttons",
        require_thinking: bool | str = True,
        default_tics: int | str = 12,
    ) -> None:
        self.image_buffers = _as_list(image_buffers)
        self.video = bool(video)
        self.video_buffer = video_buffer
        # human_view: feed the model only what a human player sees on screen
        # (first-person view + on-screen HUD), and suppress every oracle channel
        # (depth/labels/objects/sectors buffers and exact game-variable / reward /
        # step text). Privileged variables stay in the env state for logging.
        self.human_view = _as_bool(human_view)
        self.include_structured_state = _as_bool(include_structured_state) and not self.human_view
        self.include_raw_buffers = _as_bool(include_raw_buffers) and not self.human_view
        self.prompt = prompt
        self.action_format = str(action_format).lower()
        self.skill_name = skill_name
        self.require_thinking = _as_bool(require_thinking)
        self.default_tics = max(1, int(default_tics))

    def parse(self, value: Any, ctx: ParserContext) -> AgentInput:
        if not isinstance(value, EnvState):
            raise TypeError(f"VizDoomObservationParser requires EnvState, got {type(value).__name__}")
        state = value
        agent_id = ctx.agent_id
        observation = state.observation if isinstance(state.observation, dict) else {"observation": state.observation}
        content = [ContentBlock.text(self.prompt or self._prompt(observation))]

        if self.video:
            frames = observation.get(self.video_buffer)
            if frames:
                content.append(ContentBlock(type="video", data=[_buffer_to_image(frame) for frame in frames], metadata={"source": self.video_buffer}))

        for name in self.image_buffers:
            key = _buffer_key(name)
            if key in observation and observation[key] is not None:
                content.append(ContentBlock(type="image", data=_buffer_to_image(observation[key]), metadata={"source": key}))

        if self.include_structured_state:
            content.append(ContentBlock(type="vizdoom_state", data={key: value for key, value in observation.items() if key not in _VIZDOOM_BUFFER_KEYS}, metadata={"agent_id": agent_id}))
        if self.include_raw_buffers:
            for key in sorted(_VIZDOOM_BUFFER_KEYS - {"screen_history"}):
                if key in observation and key not in {_buffer_key(name) for name in self.image_buffers}:
                    content.append(ContentBlock(type=f"vizdoom_{key}", data=observation[key], metadata={"source": key}))

        return AgentInput(content=content, metadata={"env_id": state.env_id, "step_idx": state.step_idx, "agent_id": agent_id})

    def _prompt(self, observation: dict[str, Any]) -> str:
        lines = []
        instruction = observation.get("instruction")
        if instruction:
            lines.append(str(instruction))
        decision_tics = int(observation.get("decision_tics") or self.default_tics)

        if not self.human_view:
            lines.append(f"VizDoom step: {observation.get('step_idx', 0)}")
            lines.append(f"Episode time: {observation.get('episode_time', 0)}")
            lines.append(f"Total reward: {observation.get('total_reward', 0.0)}")
            history_length = observation.get("screen_history_length")
            if history_length:
                lines.append(f"Current video segment: {history_length} recent simulator frames from the last executed action.")

            game_variables = observation.get("game_variables") or {}
            tracked_variables = observation.get("tracked_game_variables") or {}
            if game_variables or tracked_variables:
                merged = {**game_variables, **tracked_variables}
                lines.append(f"Game variables: {json.dumps(merged, ensure_ascii=False, sort_keys=True)}")

            for key in ["labels", "objects", "sectors"]:
                values = observation.get(key)
                if isinstance(values, list):
                    lines.append(f"{key}: {len(values)} visible")

            notifications = observation.get("notifications_buffer")
            if notifications:
                lines.append(f"Notifications: {notifications}")

        actions = observation.get("available_buttons") or []
        action_text = ", ".join(actions) if actions else "the scenario's available buttons"
        lines.append(f"Available buttons: {action_text}.")
        if self.action_format == "json":
            lines.append('Respond with one button name, NOOP, SUBMIT, or JSON like {"buttons": ["MOVE_FORWARD", "ATTACK"], "tics": 1}.')
        else:
            lines.extend(_skill_prompt_lines(self.skill_name, action_text, require_thinking=self.require_thinking, default_tics=decision_tics))
        return "\n".join(lines)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    return [str(item).upper() for item in value]


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _buffer_key(name: str) -> str:
    normalized = str(name).lower()
    return normalized if normalized.endswith("_buffer") else f"{normalized}_buffer"


def _buffer_to_image(buffer: Any) -> Any:
    Image, has_pil = optional_import("PIL.Image")
    if not has_pil or buffer is None:
        return buffer
    if getattr(buffer.__class__, "__module__", "").startswith("PIL."):
        return buffer

    array = buffer
    if hasattr(array, "ndim") and array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = array.transpose(1, 2, 0)
    if hasattr(array, "ndim") and array.ndim == 3 and array.shape[-1] == 4:
        return Image.fromarray(array).convert("RGB")
    return Image.fromarray(array)


def _skill_prompt_lines(skill_name: str, action_text: str, *, require_thinking: bool = True, default_tics: int = 12) -> list[str]:
    # Draw the tool-call examples from the buttons that are actually available in
    # this scenario. Hard-coding "ATTACK" made the model copy a button that some
    # scenarios (e.g. take_cover, health_gathering) do not allow, producing
    # invalid actions.
    buttons = [item.strip() for item in action_text.split(",") if item.strip()]
    has_button_list = bool(buttons) and not action_text.startswith("the scenario")
    single_example = buttons[0] if has_button_list else "ATTACK"
    # A meaningful simultaneous combo only exists when ATTACK can be held together
    # with a movement/turn button (e.g. move and shoot at once). For strafe-only
    # or turn-only scenarios there is no sensible combo, so we do not advertise one
    # (and never suggest opposing buttons like MOVE_LEFT + MOVE_RIGHT).
    move_like = [b for b in buttons if b != "ATTACK"] if has_button_list else []
    combo_example = f"{move_like[0]}, ATTACK" if (has_button_list and "ATTACK" in buttons and move_like) else None

    lines = []
    if require_thinking:
        lines.extend(
            [
                "First write a concise <think>...</think> block.",
                "In <think>, inspect the current video/state, compare with recent history, and decide the single next action in 1-3 short sentences.",
                "After </think>, immediately write exactly one VizDoom skill call. Do not stop after the thinking block.",
            ]
        )
    lines.extend(
        [
            "Decide only the action for the CURRENT frame, then output exactly one skill call for that single action. The call runs for the requested tics, and you will be asked again for the next frame.",
            f"- {skill_name}(buttons, tics={default_tics}): choose your action for this frame. buttons must come from: {action_text}.",
            "Only use buttons from that list; any other button is rejected as an invalid action.",
            "Almost always pick a SINGLE button. Buttons listed together are held down at the same time (e.g. move and shoot at once) for this one frame — they are NOT a sequence of moves. Do not plan ahead, do not repeat a button, and never combine opposing buttons such as MOVE_LEFT with MOVE_RIGHT or TURN_LEFT with TURN_RIGHT.",
            f"- noop(tics={default_tics}): do nothing until the next decision.",
            "- submit(): end the rollout only when the objective is complete.",
            "Prefer Qwen tool-call format (a single button):",
            "<tool_call>",
            f"<function={skill_name}>",
            f"<parameter=buttons>{single_example}</parameter>",
            f"<parameter=tics>{default_tics}</parameter>",
            f"</function>",
            "</tool_call>",
        ]
    )
    if combo_example:
        lines.append(f"Only when you genuinely need two buttons held at once this frame, list them together, e.g. <parameter=buttons>{combo_example}</parameter>.")
    lines.append(f"If tool calls are unavailable, write {skill_name}({single_example}). Do not answer with JSON.")
    return lines


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ACTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", re.DOTALL)


class _VizDoomActionDecoder:
    """Implementation helper behind :func:`vizdoom_action_parser`."""

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


_OBSERVATION_FORMATTER = _VizDoomObservationFormatter(human_view=True, video=True, image_buffers=["screen"])


def vizdoom_observation_parser(value: Any, context: ParserContext) -> Any:
    """Convert a ViZDoom ``EnvState`` into the model-facing request value."""

    return _OBSERVATION_FORMATTER.parse(value, context)


def vizdoom_qwen_output_parser(value: Any, context: ParserContext) -> Any:
    """Normalize Qwen reasoning/tool-call wrappers before action decoding."""

    del context
    if not isinstance(value, AgentOutput):
        raise TypeError(f"vizdoom_qwen_output_parser requires AgentOutput, got {type(value).__name__}")
    text = value.first_text() or ""
    normalized_text = _strip_thinking(text)
    metadata = dict(value.metadata)
    metadata["raw_text"] = text
    metadata["normalized_text"] = normalized_text
    metadata["tool_calls"] = [*_metadata_tool_calls(metadata), *_extract_tool_calls(text)]

    content = []
    replaced_text = False
    for block in value.content:
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


def vizdoom_action_parser(value: Any, context: ParserContext) -> Any:
    """Convert a model value into a ViZDoom ``ParsedAction``."""

    observation = context.state.observation if context.state is not None else None
    buttons = observation.get("available_buttons") if isinstance(observation, dict) else None
    return _VizDoomActionDecoder(buttons=buttons).parse(value, context)


def _strip_thinking(text: str) -> str:
    candidate = text.strip()
    if "</think>" in candidate:
        after_thinking = candidate.rsplit("</think>", 1)[-1].strip()
        if after_thinking:
            return after_thinking
        return re.sub(r"</?think>", "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _metadata_tool_calls(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = metadata.get("tool_calls")
    return list(tool_calls) if isinstance(tool_calls, list) else []

from __future__ import annotations

import json
from typing import Any

from lmms_eval.agentic.parsers.base import ObservationParser, ParserContext
from lmms_eval.agentic.types import AgentInput, ContentBlock, EnvState
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


class VizDoomObservationParser(ObservationParser):
    """Convert VizDoom state into chat-friendly text, video, image, and state blocks."""

    def __init__(
        self,
        image_buffers: list[str] | str | None = ("screen",),
        video: bool = False,
        video_buffer: str = "screen_history",
        include_structured_state: bool = True,
        include_raw_buffers: bool = True,
        prompt: str | None = None,
        action_format: str = "skill",
        skill_name: str = "press_buttons",
        require_thinking: bool | str = True,
        default_tics: int | str = 12,
    ) -> None:
        self.image_buffers = _as_list(image_buffers)
        self.video = bool(video)
        self.video_buffer = video_buffer
        self.include_structured_state = include_structured_state
        self.include_raw_buffers = include_raw_buffers
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
        lines.append(f"VizDoom step: {observation.get('step_idx', 0)}")
        lines.append(f"Episode time: {observation.get('episode_time', 0)}")
        lines.append(f"Total reward: {observation.get('total_reward', 0.0)}")
        decision_tics = int(observation.get("decision_tics") or self.default_tics)
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
    lines = []
    if require_thinking:
        lines.extend(
            [
                "First write a concise <think>...</think> block.",
                "In <think>, inspect the current video/state, compare with recent history, and explain the next action in 1-3 short sentences.",
                "After </think>, immediately write exactly one VizDoom skill call. Do not stop after the thinking block.",
            ]
        )
    lines.extend(
        [
            "The skill call chooses the next action and will run for the requested tics before the next model decision.",
            f"- {skill_name}(buttons, tics={default_tics}): press one or more available buttons. buttons must come from: {action_text}.",
            f"- noop(tics={default_tics}): do nothing until the next decision.",
            "- submit(): end the rollout only when the objective is complete.",
            "Prefer Qwen tool-call format:",
            "<tool_call>",
            f"<function={skill_name}>",
            "<parameter=buttons>ATTACK</parameter>",
            f"<parameter=tics>{default_tics}</parameter>",
            f"</function>",
            "</tool_call>",
            f"For multiple buttons, use <parameter=buttons>MOVE_LEFT, ATTACK</parameter>. If tool calls are unavailable, write {skill_name}(ATTACK). Do not answer with JSON.",
        ]
    )
    return lines

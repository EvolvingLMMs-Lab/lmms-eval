"""Generic observation rendering for environments using reserved keys."""

from __future__ import annotations

import json
import re
from typing import Any

from lmms_eval.agentic.parsers.base import ObservationParser, ParserContext
from lmms_eval.agentic.types import ActionSpec, AgentInput, ContentBlock, EnvState


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


def _has_frames(value: Any) -> bool:
    if value is None:
        return False
    length = getattr(value, "__len__", None)
    return bool(len(value)) if callable(length) else True


def _as_flag(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

"""Prompt construction and action decoding belong to the MiniGrid task."""

import json
import re

from PIL import Image

from lmms_eval.tasks.minigrid_agentic.env import ACTION_DESCRIPTIONS, ACTION_INDICES

ALIASES = {"TURN_LEFT": "LEFT", "TURN_RIGHT": "RIGHT", "MOVE_FORWARD": "FORWARD", "GO_FORWARD": "FORWARD", "PICK_UP": "PICKUP", "OPEN": "TOGGLE", "USE": "TOGGLE"}


def observation_messages(doc, state=None, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    state = state or {}
    sections = [
        str(kwargs.get("pre_prompt", "")) + str(doc.get("instruction", "")) + str(kwargs.get("post_prompt", "")),
        str(state.get("mission", "")),
        f"Variables: {json.dumps(state['variables'], sort_keys=True)}" if state.get("variables") else "",
        f"Step {int(state.get('metrics', {}).get('steps', 0))}.",
        "Available actions:\n" + "\n".join(f"- {name}: {description}" for name, description in ACTION_DESCRIPTIONS.items()),
        "Respond with only the action name.",
    ]
    content = [{"type": "text", "text": "\n\n".join(section for section in sections if section)}]
    image = state.get("image")
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        content.append({"type": "image", "url": image})
    return [{"role": "user", "content": content}]


def parse_action(text):
    if not isinstance(text, str):
        return None
    # Some chat templates prefill the opening tag, so only </think> is returned.
    text = text.rsplit("</think>", 1)[-1].strip()
    valid = set(ACTION_INDICES)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            mapping, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(mapping, dict):
            continue
        candidates = [mapping.get(key) for key in ("action", "action_name", "name", "tool_name", "type")]
        if isinstance(mapping.get("arguments"), dict):
            candidates.extend(mapping["arguments"].get(key) for key in ("action", "action_name", "name", "type"))
        for candidate in candidates:
            if isinstance(candidate, str):
                name = candidate.strip().upper()
                name = ALIASES.get(name, name)
                if name in valid:
                    return name
    for candidate in sorted(valid | set(ALIASES), key=lambda name: (-len(name), name)):
        if re.search(rf"\b{re.escape(candidate)}\b", text.upper()):
            return ALIASES.get(candidate, candidate)
    return None

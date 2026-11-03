"""ViZDoom prompts and action decoding; no framework parser registry."""

from __future__ import annotations

import json
import math
import re


def vizdoom_messages(observation):
    from PIL import Image

    buttons = observation.get("available_buttons", [])
    prompt = "\n".join(
        [
            observation.get("instruction", ""),
            "The images show the recent simulator frames in time order. Choose the single next action.",
            f"Available buttons: {', '.join(buttons)}.",
            f"Reply with press_buttons(BUTTON, tics={observation.get('decision_tics', 5)}), noop(), or submit().",
            "Buttons in one call are held simultaneously. Never combine opposing buttons such as MOVE_LEFT and MOVE_RIGHT or TURN_LEFT and TURN_RIGHT.",
            "Use only the listed buttons. Submit only when the objective is complete.",
        ]
    )
    content = [{"type": "text", "text": prompt}]
    for frame in observation.get("frames", []):
        if not isinstance(frame, Image.Image):
            if frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
                frame = frame.transpose(1, 2, 0)
            frame = Image.fromarray(frame)
        content.append({"type": "image", "url": frame.convert("RGB")})
    return [{"role": "user", "content": content}]


def parse_vizdoom_action(text, available_buttons):
    """Accept tool calls, ordinary calls, JSON, button names and short prose.

    All formats produce the same small dict. Unknown or malformed actions have
    buttons=None and consume an invalid turn in the environment.
    """
    text = str(text).rsplit("</think>", 1)[-1].strip()
    available = set(available_buttons)
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE):
        payload = match.group(1)
        function = re.search(r"<function=([^>\s]+)>", payload, re.IGNORECASE)
        if function:
            args = {key.lower(): value.strip() for key, value in re.findall(r"<parameter=([^>\s]+)>(.*?)</parameter>", payload, re.DOTALL | re.IGNORECASE)}
            action = _call_action(function.group(1), args, available)
            if action is not None:
                return action
    for name, arguments in re.findall(r"\b([A-Za-z_]\w*)\s*\((.*?)\)", text, re.DOTALL):
        args = {"buttons": arguments}
        args.update({key.lower(): value.strip().strip("\"'") for key, value in re.findall(r"\b(\w+)\s*=\s*([^,)]+)", arguments)})
        action = _call_action(name, args, available)
        if action is not None:
            return action
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group())
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if payload.get("submit") is True:
                return {"submit": True}
            action = _call_action(str(payload.get("action", "press_buttons")), payload, available)
            if action is not None:
                return action
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.upper())
    if "SUBMIT" in tokens:
        return {"submit": True}
    if "NOOP" in tokens:
        return {"buttons": {}}
    buttons = list(dict.fromkeys(token for token in tokens if token in available))
    if not buttons:
        for name, pattern in {
            "ATTACK": r"\b(SHOOT|SHOOTING|FIRE|FIRING)\b",
            "MOVE_LEFT": r"\b(MOVE|STRAFE|GO|TURN)\s+LEFT\b",
            "MOVE_RIGHT": r"\b(MOVE|STRAFE|GO|TURN)\s+RIGHT\b",
        }.items():
            if name in available and re.search(pattern, text.upper()):
                buttons.append(name)
    return {"buttons": {name: 1.0 for name in buttons} if buttons else None}


def _call_action(name, args, available):
    name = name.upper()
    if name == "SUBMIT":
        return {"submit": True}
    if name == "NOOP":
        return {"buttons": {}, **_tics(args)}
    if name in available:
        return {"buttons": {name: 1.0}, **_tics(args)}
    if name not in {"PRESS_BUTTONS", "PRESS_BUTTON", "VIZDOOM_PRESS", "VIZDOOM_ACTION", "ACT"}:
        return None
    value = args.get("buttons", args.get("button", args.get("actions")))
    if isinstance(value, str):
        # Discard keyword assignments before extracting button tokens.
        value = re.sub(r"\b\w+\s*=\s*[^,]+", "", value)
        value = re.findall(r"[A-Za-z][A-Za-z0-9_]*", value.upper())
    if isinstance(value, (list, tuple)):
        value = {str(button).upper(): 1.0 for button in value}
    if not isinstance(value, dict) or not value:
        return None
    try:
        buttons = {str(button).upper(): float(amount) for button, amount in value.items()}
    except (TypeError, ValueError):
        return {"buttons": None}
    if any(name not in available or not math.isfinite(amount) for name, amount in buttons.items()):
        return {"buttons": None}
    return {"buttons": buttons, **_tics(args)}


def _tics(args):
    try:
        return {"tics": max(1, min(35, int(args["tics"])))} if "tics" in args else {}
    except (TypeError, ValueError, OverflowError):
        return {}

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
SUBMIT_PATTERN = re.compile(r"<submit>(.*?)</submit>", re.DOTALL)

_EPISODES: dict[str, "VizDoomEpisode"] = {}


@dataclass
class VizDoomEpisode:
    episode_id: str
    game: Any
    action_map: dict[str, list[int]]
    output_dir: Path
    video_fps: int
    observation_frames: int
    frames: list[np.ndarray] = field(default_factory=list)
    total_reward: float = 0.0
    steps: int = 0
    closed: bool = False
    last_observation_video_path: str | None = None
    final_video_path: str | None = None


def vizdoom_doc_to_target(doc):
    return doc.get("target_state", {})


def vizdoom_doc_to_visual(doc):
    episode = _get_or_create_episode(doc)
    return [episode.last_observation_video_path] if episode.last_observation_video_path else []


def vizdoom_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
    if round_idx is None:
        return _build_prompt(doc, _initial_state_for_prompt(doc), None)

    episode = _get_or_create_episode(doc)
    state_info = previous_round_info or _initial_round_info(doc, episode)
    model_response = previous_output[-1] if previous_output else ""

    submit_payloads = _extract_tag_payload(SUBMIT_PATTERN, model_response)
    if submit_payloads and (episode.steps > 0 or episode.game.is_episode_finished()):
        final_payload = _finalize_episode(doc, episode, state_info, "model_submit", previous_output, submit_payloads[-1])
        return None, None, True, [json.dumps(final_payload, ensure_ascii=False)], state_info
    if submit_payloads:
        state_info["invalid_actions"] = state_info.get("invalid_actions", 0) + 1
        state_info["last_result"] = {
            "error": "submit rejected because the episode has not started",
            "hint": _action_format_hint(doc),
        }
        return _next_turn(doc, episode, state_info, previous_output)

    tool_payloads = _extract_action_payloads(model_response)
    if len(tool_payloads) != 1:
        state_info["invalid_actions"] = state_info.get("invalid_actions", 0) + 1
        state_info["last_result"] = {
            "error": "expected exactly one <tool_call>",
            "parsed_tool_calls": len(tool_payloads),
            "hint": _action_format_hint(doc),
        }
        if _should_stop(doc, episode, round_idx):
            final_payload = _finalize_episode(doc, episode, state_info, "step_limit", previous_output)
            return None, None, True, [json.dumps(final_payload, ensure_ascii=False)], state_info
        return _next_turn(doc, episode, state_info, previous_output)

    tool_payload = tool_payloads[0]
    arguments = tool_payload.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    action_name = str(arguments.get("action", tool_payload.get("action", ""))).strip().upper()
    if not action_name and str(tool_payload.get("name", "")).strip().upper() in _allowed_actions_for_prompt(doc):
        action_name = str(tool_payload.get("name", "")).strip().upper()
    action_name = action_name.replace(" ", "_")
    state_info.setdefault("actions", []).append(action_name)

    if action_name not in episode.action_map:
        state_info["invalid_actions"] = state_info.get("invalid_actions", 0) + 1
        state_info["last_result"] = {
            "error": f"unknown or unavailable action '{action_name}'",
            "available_actions": sorted(episode.action_map),
        }
        if _should_stop(doc, episode, round_idx):
            final_payload = _finalize_episode(doc, episode, state_info, "step_limit", previous_output)
            return None, None, True, [json.dumps(final_payload, ensure_ascii=False)], state_info
        return _next_turn(doc, episode, state_info, previous_output)

    step_result = _step_episode(doc, episode, action_name)
    state_info["valid_actions"] = state_info.get("valid_actions", 0) + 1
    state_info["last_action"] = action_name
    state_info["last_result"] = step_result
    state_info["state"] = _episode_state(episode)

    if step_result.get("done") or _should_stop(doc, episode, round_idx):
        reason = "game_finished" if step_result.get("done") else "step_limit"
        final_payload = _finalize_episode(doc, episode, state_info, reason, previous_output)
        return None, None, True, [json.dumps(final_payload, ensure_ascii=False)], state_info

    return _next_turn(doc, episode, state_info, previous_output)


def vizdoom_process_results(doc, results):
    raw = results[0] if results else ""
    success = 0.0
    total_reward = 0.0
    episode_steps = 0.0
    invalid_actions = 0.0
    valid_action_rate = 0.0

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            success = 1.0 if payload.get("success") else 0.0
            total_reward = float(payload.get("total_reward", 0.0))
            episode_steps = float(payload.get("episode_steps", 0.0))
            invalid_actions = float(payload.get("invalid_actions", 0.0))
            valid_actions = float(payload.get("valid_actions", 0.0))
            valid_action_rate = valid_actions / max(valid_actions + invalid_actions, 1.0)
            if payload.get("video_path"):
                eval_logger.info("ViZDoom episode video: {}", payload["video_path"])
        except json.JSONDecodeError:
            pass

    return {
        "vizdoom_success": success,
        "vizdoom_total_reward": total_reward,
        "vizdoom_episode_steps": episode_steps,
        "vizdoom_valid_action_rate": valid_action_rate,
        "vizdoom_invalid_actions": invalid_actions,
    }


def vizdoom_aggregate_mean(results):
    if not results:
        return 0.0
    return sum(results) / len(results)


def _extract_tag_payload(pattern, text):
    matches = pattern.findall(text or "")
    payloads = []
    for match in matches:
        candidate = match.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _extract_action_payloads(text):
    payloads = _extract_tag_payload(TOOL_CALL_PATTERN, text)
    if payloads:
        return payloads

    for obj_text in _extract_braced_objects(text):
        try:
            payload = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("name") or payload.get("action") or payload.get("arguments"):
            payloads.append(payload)
    return payloads


def _extract_braced_objects(text):
    objects = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(text or ""):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : idx + 1])
                start = -1

    return objects


def _initial_state_for_prompt(doc):
    return {
        "episode_id": _episode_id(doc),
        "scenario_config": doc.get("scenario_config", "basic.cfg"),
        "seed": doc.get("seed"),
        "step": 0,
        "total_reward": 0.0,
        "max_episode_steps": int(doc.get("max_episode_steps", 12)),
    }


def _initial_round_info(doc, episode):
    return {
        "episode_id": episode.episode_id,
        "state": _episode_state(episode),
        "valid_actions": 0,
        "invalid_actions": 0,
        "actions": [],
        "last_action": None,
        "last_result": None,
    }


def _build_prompt(doc, state, tool_result):
    allowed_actions = _ordered_actions_for_prompt(doc)
    valid_replies = "\n".join(
        f'<tool_call>{{"name":"act","arguments":{{"action":"{action}"}}}}</tool_call>' for action in allowed_actions
    )
    result_text = ""
    if tool_result is not None:
        result_text = f"\nLast action result: {json.dumps(tool_result, ensure_ascii=False)}"

    return (
        f"{doc.get('user_query', 'Play one ViZDoom episode.')}\n"
        f"Current episode state: {json.dumps(state, ensure_ascii=False)}{result_text}\n"
        "You are controlling a ViZDoom agent from the attached short video observation.\n"
        "Use the video to decide whether the enemy is centered before firing.\n"
        "Policy: choose MOVE_LEFT or MOVE_RIGHT to align the enemy with the center/crosshair; choose ATTACK only when the enemy is already centered or very close to centered.\n"
        "If the last ATTACK gave no positive progress, choose a movement action before attacking again.\n"
        "The evaluator will stop the episode automatically; do not send a termination message, final answer, or natural language.\n"
        "Output format is mandatory: do not answer with only the action word.\n"
        "Your entire response must be exactly one complete line copied from these valid tool calls:\n"
        f"{valid_replies}"
    )


def _action_format_hint(doc):
    default_action = "ATTACK" if "ATTACK" in _allowed_actions_for_prompt(doc) else _allowed_actions_for_prompt(doc)[0]
    return f'<tool_call>{{"name":"act","arguments":{{"action":"{default_action}"}}}}</tool_call>'


def _allowed_actions_for_prompt(doc):
    actions = doc.get("allowed_actions") or ["ATTACK", "MOVE_LEFT", "MOVE_RIGHT", "NOOP"]
    normalized = [str(action).strip().upper().replace(" ", "_") for action in actions if str(action).strip()]
    return normalized or ["NOOP"]


def _ordered_actions_for_prompt(doc):
    actions = _allowed_actions_for_prompt(doc)
    movement_first = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "NOOP"]
    ordered = [action for action in movement_first if action in actions]
    ordered.extend(action for action in actions if action not in ordered)
    return ordered


def _next_turn(doc, episode, state_info, previous_output):
    next_prompt = _build_prompt(doc, state_info.get("state", _episode_state(episode)), state_info.get("last_result"))
    visuals = [episode.last_observation_video_path] if episode.last_observation_video_path else []
    return visuals, next_prompt, False, previous_output, state_info


def _step_episode(doc, episode, action_name):
    game = episode.game
    reward = 0.0
    done = False
    frame_skip = max(1, int(doc.get("frame_skip", 4)))

    for _ in range(frame_skip):
        if game.is_episode_finished():
            done = True
            break
        reward += float(game.make_action(episode.action_map[action_name], 1))
        episode.steps += 1
        _capture_current_frame(episode)
        if game.is_episode_finished():
            done = True
            break

    episode.total_reward += reward
    _write_observation_video(episode)
    state = _episode_state(episode)
    return {
        "ok": True,
        "action": action_name,
        "reward": reward,
        "total_reward": episode.total_reward,
        "done": done,
        "state": state,
        "observation_video_path": episode.last_observation_video_path,
    }


def _should_stop(doc, episode, round_idx):
    max_steps = int(doc.get("max_episode_steps", 12))
    return bool(episode.game.is_episode_finished() or round_idx >= max_steps)


def _finalize_episode(doc, episode, state_info, terminal_reason, previous_output, submit=None):
    if not episode.final_video_path:
        episode.final_video_path = _write_video_artifact(episode, "episode")

    state = _episode_state(episode)
    target = doc.get("target_state", {})
    min_reward = float(target.get("min_total_reward", 0.0))
    success = episode.total_reward >= min_reward

    valid_actions = int(state_info.get("valid_actions", 0))
    invalid_actions = int(state_info.get("invalid_actions", 0))
    payload = {
        "success": success,
        "terminal_reason": terminal_reason,
        "total_reward": episode.total_reward,
        "episode_steps": episode.steps,
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
        "state": state,
        "submit": submit,
        "trace": previous_output or [],
        "actions": state_info.get("actions", []),
        "video_path": episode.final_video_path,
        "observation_video_path": episode.last_observation_video_path,
    }
    state_info["state"] = state
    state_info["terminal_reason"] = terminal_reason
    state_info["video_path"] = episode.final_video_path
    _close_episode(episode.episode_id)
    return payload


def _get_or_create_episode(doc):
    episode_id = _episode_id(doc)
    episode = _EPISODES.get(episode_id)
    if episode is not None and not episode.closed:
        return episode

    game = _new_game(doc)
    output_dir = _episode_output_dir(episode_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = VizDoomEpisode(
        episode_id=episode_id,
        game=game,
        action_map=_build_action_map(game, _allowed_actions_for_prompt(doc)),
        output_dir=output_dir,
        video_fps=int(doc.get("video_fps", 8)),
        observation_frames=int(doc.get("observation_frames", 8)),
    )
    _capture_current_frame(episode)
    _write_observation_video(episode)
    _EPISODES[episode_id] = episode
    return episode


def _new_game(doc):
    try:
        import vizdoom
    except ImportError as exc:
        raise ImportError("The vizdoom_agent task requires the optional `vizdoom` package. Install it with `uv pip install vizdoom`.") from exc

    game = vizdoom.DoomGame()
    scenario_path = _resolve_scenario_path(vizdoom, doc.get("scenario_config", "basic.cfg"))
    game.load_config(str(scenario_path))
    game.set_window_visible(False)
    if hasattr(vizdoom, "Mode"):
        game.set_mode(vizdoom.Mode.PLAYER)
    if hasattr(vizdoom, "ScreenFormat"):
        game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    if hasattr(vizdoom, "ScreenResolution"):
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_320X240)
    if doc.get("episode_timeout") is not None:
        game.set_episode_timeout(int(doc["episode_timeout"]))
    if doc.get("seed") is not None and hasattr(game, "set_seed"):
        game.set_seed(int(doc["seed"]))
    game.init()
    game.new_episode()
    return game


def _resolve_scenario_path(vizdoom, scenario_config):
    scenario = Path(str(scenario_config))
    candidates = []
    if scenario.is_absolute():
        candidates.append(scenario)
    else:
        candidates.append(Path.cwd() / scenario)
        env_dir = os.getenv("LMMS_EVAL_VIZDOOM_SCENARIO_DIR")
        if env_dir:
            candidates.append(Path(env_dir) / scenario)
        package_scenarios = getattr(vizdoom, "scenarios_path", None)
        if callable(package_scenarios):
            package_scenarios = package_scenarios()
        if package_scenarios:
            candidates.append(Path(package_scenarios) / scenario)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find ViZDoom scenario config '{scenario_config}'. Checked: {checked}")


def _build_action_map(game, allowed_actions):
    available_buttons = [str(button).split(".")[-1].upper() for button in game.get_available_buttons()]
    action_map = {"NOOP": [0] * len(available_buttons)}

    for action_name in allowed_actions:
        if action_name == "NOOP":
            continue
        if action_name not in available_buttons:
            continue
        action = [0] * len(available_buttons)
        action[available_buttons.index(action_name)] = 1
        action_map[action_name] = action
    return action_map


def _capture_current_frame(episode):
    state = episode.game.get_state()
    if state is None or state.screen_buffer is None:
        return
    episode.frames.append(_normalize_frame(state.screen_buffer))


def _normalize_frame(frame):
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _write_observation_video(episode):
    recent = episode.frames[-episode.observation_frames :]
    if not recent:
        return None
    episode.last_observation_video_path = _write_video_artifact(episode, f"observation_step_{episode.steps:04d}", recent)
    return episode.last_observation_video_path


def _write_video_artifact(episode, stem, frames=None):
    frames = frames if frames is not None else episode.frames
    if not frames:
        return None
    if len(frames) == 1:
        frames = [frames[0], frames[0]]

    mp4_path = episode.output_dir / f"{stem}.mp4"
    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), episode.video_fps, (width, height))
        if writer.isOpened():
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return str(mp4_path)
        writer.release()
    except Exception as exc:
        eval_logger.warning("Failed to write ViZDoom mp4 '{}': {}", mp4_path, exc)

    gif_path = episode.output_dir / f"{stem}.gif"
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=max(1, int(1000 / max(1, episode.video_fps))), loop=0)
    return str(gif_path)


def _episode_state(episode):
    state = {
        "step": episode.steps,
        "total_reward": episode.total_reward,
        "finished": bool(episode.game.is_episode_finished()),
    }
    for name in ["HEALTH", "AMMO2", "KILLCOUNT", "FRAGCOUNT"]:
        value = _try_game_variable(episode.game, name)
        if value is not None:
            state[name.lower()] = value
    return state


def _try_game_variable(game, name):
    try:
        import vizdoom

        variable = getattr(vizdoom.GameVariable, name)
        return float(game.get_game_variable(variable))
    except Exception:
        return None


def _episode_id(doc):
    raw = str(doc.get("id") or f"vizdoom_seed_{doc.get('seed', int(time.time()))}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _episode_output_dir(episode_id):
    root = Path(os.getenv("LMMS_EVAL_VIZDOOM_OUTPUT_DIR", "outputs/vizdoom_agent"))
    return root / episode_id


def _close_episode(episode_id):
    episode = _EPISODES.pop(episode_id, None)
    if episode is None or episode.closed:
        return
    try:
        episode.game.close()
    finally:
        episode.closed = True

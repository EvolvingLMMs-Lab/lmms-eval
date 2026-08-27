"""Per-episode artifacts: summary.md, actions.jsonl, rollout.mp4, segments/.

Written under ``<output_path>/agentic_artifacts/<task>_doc<id>_<timestamp>/``
whenever the eval runs with ``--output_path``. Video writing needs ``av`` and
``numpy``; failures are recorded next to the artifacts instead of aborting the
rollout.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from lmms_eval.agentic.trace import (
    action_label,
    info_without_frames,
    payload_to_compact_trace,
    safe_data,
)
from lmms_eval.agentic.types import EnvState, EpisodeResult, EpisodeStep, GameAction
from lmms_eval.imports import optional_import


def write_episode_artifacts(result: EpisodeResult, *, output_path: str | None, task_name: str, doc_id: int) -> dict[str, str]:
    if not output_path:
        return {}

    artifact_dir = _new_artifact_dir(Path(output_path), task_name=task_name, doc_id=doc_id)
    rows = _episode_action_rows(result)
    artifacts: dict[str, str] = {}

    summary_path = artifact_dir / "summary.md"
    summary_path.write_text(_episode_summary_markdown(result, rows), encoding="utf-8")
    artifacts["summary"] = str(summary_path)

    actions_path = artifact_dir / "actions.jsonl"
    with actions_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    artifacts["actions"] = str(actions_path)

    frames = _episode_video_frames(result)
    if frames:
        video_path = artifact_dir / "rollout.mp4"
        try:
            _write_mp4(video_path, frames, fps=_artifact_fps())
            artifacts["video"] = str(video_path)
        except Exception as exc:
            error_path = artifact_dir / "video_error.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            artifacts["video_error"] = str(error_path)

    if _write_action_segments(artifact_dir, result):
        artifacts["segments"] = str(artifact_dir / "segments")

    return artifacts


def _write_action_segments(artifact_dir: Path, result: EpisodeResult) -> int:
    """Write one mp4 per action (captured intra-action frames). Requires the env's emit_action_frames."""

    segments = [(idx, _step_action_frames(step)) for idx, step in enumerate(result.steps)]
    segments = [(idx, segment) for idx, segment in segments if segment]
    if not segments:
        return 0

    segments_dir = artifact_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    fps = _artifact_fps()
    for idx, segment in segments:
        try:
            _write_mp4(segments_dir / f"step_{idx:03d}.mp4", segment, fps=fps)
            written += 1
        except Exception as exc:
            (segments_dir / f"step_{idx:03d}_error.txt").write_text(str(exc), encoding="utf-8")
    return written


def _new_artifact_dir(output_path: Path, *, task_name: str, doc_id: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_filename(f"{task_name}_doc{doc_id}_{timestamp}")
    base = output_path / "agentic_artifacts"
    artifact_dir = base / stem
    suffix = 1
    while artifact_dir.exists():
        artifact_dir = base / f"{stem}_{suffix}"
        suffix += 1
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _episode_action_rows(result: EpisodeResult) -> list[dict[str, Any]]:
    rows = []
    for step in result.steps:
        info = step.result.info if step.result is not None and isinstance(step.result.info, dict) else {}
        action = step.parsed_action.action if step.parsed_action is not None else None
        requested_action = action_label(action)
        rows.append(
            {
                "step_idx": step.state.step_idx,
                "model_output": payload_to_compact_trace(step.output),
                "raw_model_output": payload_to_compact_trace(step.raw_output),
                "requested_action": requested_action,
                "executed_action": _executed_action_label(info, fallback=requested_action),
                "action_data": safe_data(action.data) if isinstance(action, GameAction) else safe_data(action),
                "parse_error": step.parsed_action.error if step.parsed_action is not None else None,
                "env_error": safe_data(info.get("error")),
                "invalid_actions": safe_data(info.get("invalid_actions")),
                "reward": safe_data(step.result.reward) if step.result is not None else None,
                "total_reward": safe_data(info.get("total_reward")),
                "done": step.result.done if step.result is not None else None,
                "info": safe_data(info_without_frames(info)),
            }
        )
    return rows


def _episode_summary_markdown(result: EpisodeResult, rows: list[dict[str, Any]]) -> str:
    requested_counts = Counter(row["requested_action"] for row in rows)
    executed_counts = Counter(row["executed_action"] for row in rows)
    lines = [
        "# Agentic Rollout Summary",
        "",
        f"- Success: {result.success}",
        f"- Steps: {len(rows)}",
        f"- Final terminal: {result.final_state.terminal}",
        f"- Metrics: `{json.dumps(safe_data(result.metrics or {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Requested action counts: `{json.dumps(dict(requested_counts), ensure_ascii=False, sort_keys=True)}`",
        f"- Executed action counts: `{json.dumps(dict(executed_counts), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Steps",
        "",
        "| step | requested | executed | reward | total_reward | done | env_error | model_output |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        cells = [row["step_idx"], row["requested_action"], row["executed_action"], row["reward"], row["total_reward"], row["done"], row["env_error"], row["model_output"]]
        lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _episode_video_frames(result: EpisodeResult) -> list[Any]:
    frames = []
    for step in result.steps:
        segment = _step_action_frames(step)
        if segment:
            frames.extend(segment)
            continue
        frame = _state_screen_frame(step.state)
        if frame is not None:
            frames.append(frame)
    final_frame = _state_screen_frame(result.final_state)
    if final_frame is not None:
        frames.append(final_frame)
    return frames


def _step_action_frames(step: EpisodeStep) -> list[Any]:
    info = step.result.info if step.result is not None else None
    frames = info.get("action_frames") if isinstance(info, dict) else None
    return list(frames) if isinstance(frames, list) else []


def _state_screen_frame(state: EnvState) -> Any:
    observation = state.observation if isinstance(state.observation, dict) else {}
    for key in ("screen_buffer", "frame", "image"):
        frame = observation.get(key)
        if frame is not None:
            return frame
    return None


def _executed_action_label(info: dict[str, Any], *, fallback: str) -> str:
    buttons = info.get("buttons")
    if isinstance(buttons, dict):
        active = [name for name, value in buttons.items() if value]
        return "+".join(active) if active else "NOOP"
    return fallback


def _write_mp4(path: Path, frames: list[Any], *, fps: int) -> None:
    av, has_av = optional_import("av")
    if not has_av:
        raise ImportError("PyAV is required to write MP4 rollout artifacts")

    rgb_frames = [_to_rgb_array(frame) for frame in frames]
    if not rgb_frames:
        raise ValueError("No frames available for rollout video")

    height, width = rgb_frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame in rgb_frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _to_rgb_array(frame: Any):
    np, has_np = optional_import("numpy")
    if not has_np:
        raise ImportError("NumPy is required to write rollout video artifacts")

    if getattr(frame.__class__, "__module__", "").startswith("PIL."):
        array = np.asarray(frame.convert("RGB"))
    else:
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = array.transpose(1, 2, 0)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[-1] == 4:
            array = array[:, :, :3]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected an RGB-like frame, got shape {getattr(array, 'shape', None)}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    scale = _artifact_scale()
    if scale > 1:
        array = np.repeat(np.repeat(array, scale, axis=0), scale, axis=1)
    return np.ascontiguousarray(array)


def _artifact_scale() -> int:
    try:
        return max(1, int(os.getenv("LMMS_AGENTIC_ARTIFACT_SCALE", "4")))
    except ValueError:
        return 4


def _artifact_fps() -> int:
    try:
        return max(1, int(os.getenv("LMMS_AGENTIC_ARTIFACT_FPS", "12")))
    except ValueError:
        return 12


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)

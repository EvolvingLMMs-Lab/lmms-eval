from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lmms_eval.agentic.model_server import ModelServer, RolloutJob
from lmms_eval.agentic.registry import (
    build_action_parser,
    build_game_env,
    build_loop_worker,
    build_model_output_parser,
    build_model_server,
    build_observation_parser,
)
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    EpisodeResult,
    EpisodeStep,
    GameAction,
    ParsedAction,
    StepResult,
)
from lmms_eval.api.instance import Instance
from lmms_eval.imports import optional_import
from lmms_eval.utils import simple_parse_args_string


def run_generate_until_game(lm: Any, requests: list[Instance], response_cache: Any = None, cli_args: Any = None) -> list[str]:
    trace_mode = getattr(cli_args, "agentic_trace_mode", "basic") if cli_args is not None else "basic"
    plans = [_rollout_plan_from_request(index, lm, req, cli_args) for index, req in enumerate(requests)]
    outputs: list[str | None] = [None] * len(plans)
    for group in _group_plans_by_model_server(plans):
        model_server_spec = group[0].model_server_spec
        model_server = build_model_server(
            model_server_spec,
            lm=lm,
            generation_kwargs={},
            response_cache=response_cache,
        )
        jobs = [_rollout_job_from_plan(local_index, plan) for local_index, plan in enumerate(group)]
        episodes = model_server.run_rollouts(jobs)
        for plan, episode in zip(group, episodes, strict=True):
            artifacts = _write_episode_artifacts(episode, output_path=getattr(cli_args, "output_path", None), task_name=plan.task_name, doc_id=plan.doc_id)
            if artifacts:
                episode.metadata = {**episode.metadata, "artifacts": artifacts}
            outputs[plan.index] = _episode_to_json(episode, trace_mode=trace_mode)

    return [output if output is not None else "" for output in outputs]


@dataclass(slots=True)
class _RolloutPlan:
    index: int
    req: Instance
    doc: Any
    generation_kwargs: dict[str, Any]
    max_steps: int
    seed: int | None
    model_server_spec: Any
    loop_worker_spec: Any
    game_env: Any
    observation_parser: Any
    model_output_parser_spec: Any
    action_parser: Any
    lmms_eval_specific_kwargs: dict[str, Any]
    doc_id: int
    task_name: str
    split: str


def _rollout_plan_from_request(index: int, lm: Any, req: Instance, cli_args: Any) -> _RolloutPlan:
    if len(req.args) == 10:
        ctx, generation_kwargs, _doc_to_visual, game_env, observation_parser, action_parser, lmms_eval_specific_kwargs, doc_id, task_name, split = req.args
        model_server_spec = "lmms"
        loop_worker_spec = "simple"
        model_output_parser_spec = "identity"
    elif len(req.args) == 12:
        (
            ctx,
            generation_kwargs,
            _doc_to_visual,
            model_server_spec,
            loop_worker_spec,
            game_env,
            observation_parser,
            action_parser,
            lmms_eval_specific_kwargs,
            doc_id,
            task_name,
            split,
        ) = req.args
        model_output_parser_spec = "identity"
    else:
        (
            ctx,
            generation_kwargs,
            _doc_to_visual,
            model_server_spec,
            loop_worker_spec,
            game_env,
            observation_parser,
            model_output_parser_spec,
            action_parser,
            lmms_eval_specific_kwargs,
            doc_id,
            task_name,
            split,
        ) = req.args
    del ctx

    model_server_spec = _runtime_component_spec(cli_args, "agentic_model_server", "agentic_model_server_args", model_server_spec)
    loop_worker_spec = _runtime_component_spec(cli_args, "agentic_loop_worker", "agentic_loop_worker_args", loop_worker_spec)
    model_output_parser_spec = _runtime_component_spec(cli_args, "agentic_model_output_parser", "agentic_model_output_parser_args", model_output_parser_spec)

    gen_kwargs = dict(generation_kwargs or {})
    max_steps = int(gen_kwargs.pop("max_game_steps", gen_kwargs.pop("max_agentic_steps", 32)))
    seed = gen_kwargs.pop("game_seed", None)
    return _RolloutPlan(
        index=index,
        req=req,
        doc=lm.task_dict[task_name][split][doc_id],
        generation_kwargs=gen_kwargs,
        max_steps=max_steps,
        seed=seed,
        model_server_spec=model_server_spec,
        loop_worker_spec=loop_worker_spec,
        game_env=game_env,
        observation_parser=observation_parser,
        model_output_parser_spec=model_output_parser_spec,
        action_parser=action_parser,
        lmms_eval_specific_kwargs=lmms_eval_specific_kwargs,
        doc_id=int(doc_id),
        task_name=str(task_name),
        split=str(split),
    )


def _group_plans_by_model_server(plans: list[_RolloutPlan]) -> list[list[_RolloutPlan]]:
    groups: dict[str, list[_RolloutPlan]] = {}
    for plan in plans:
        groups.setdefault(_model_server_spec_key(plan.model_server_spec), []).append(plan)
    return list(groups.values())


def _model_server_spec_key(spec: Any) -> str:
    return json.dumps(_safe_data(spec), ensure_ascii=False, sort_keys=True, default=str)


def _rollout_job_from_plan(index: int, plan: _RolloutPlan) -> RolloutJob:
    def make_session(model_server: ModelServer):
        worker = _build_worker_for_plan(plan, model_server)
        new_session = getattr(worker, "new_session", None)
        return new_session(plan.doc, seed=plan.seed) if callable(new_session) else None

    def run_serial(model_server: ModelServer):
        return _build_worker_for_plan(plan, model_server).run(plan.doc, seed=plan.seed)

    return RolloutJob(index=index, make_session=make_session, run_serial=run_serial)


def _build_worker_for_plan(plan: _RolloutPlan, model_server: ModelServer):
    return build_loop_worker(
        plan.loop_worker_spec,
        model_server=model_server,
        env=build_game_env(plan.game_env, doc=plan.doc, lmms_eval_specific_kwargs=plan.lmms_eval_specific_kwargs),
        observation_parser=build_observation_parser(plan.observation_parser, doc=plan.doc, lmms_eval_specific_kwargs=plan.lmms_eval_specific_kwargs),
        model_output_parser=build_model_output_parser(plan.model_output_parser_spec, doc=plan.doc, lmms_eval_specific_kwargs=plan.lmms_eval_specific_kwargs),
        action_parser=build_action_parser(plan.action_parser, doc=plan.doc, lmms_eval_specific_kwargs=plan.lmms_eval_specific_kwargs),
        max_steps=plan.max_steps,
        generation_kwargs=plan.generation_kwargs,
        request_metadata={
            "lmms_eval": {
                "doc_id": plan.doc_id,
                "task_name": plan.task_name,
                "split": plan.split,
                "request_metadata": plan.req.metadata,
            }
        },
    )


def _runtime_component_spec(cli_args: Any, name_attr: str, args_attr: str, default_spec: Any) -> Any:
    if cli_args is None:
        return default_spec

    name = getattr(cli_args, name_attr, None)
    args = getattr(cli_args, args_attr, "")
    kwargs = _parse_runtime_args(args)
    if name is None and not kwargs:
        return default_spec

    if name is None:
        if isinstance(default_spec, dict):
            name = default_spec.get("name") or default_spec.get("type") or default_spec.get("id")
        else:
            name = default_spec

    if not kwargs:
        return name
    if isinstance(name, dict):
        spec = dict(name)
    else:
        spec = {"name": name}
    spec.update(kwargs)
    return spec


def _parse_runtime_args(args: Any) -> dict[str, Any]:
    kwargs = simple_parse_args_string(args) if isinstance(args, str) else dict(args or {})
    return {key: _decode_json_value(value) for key, value in kwargs.items()}


def _decode_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    candidate = value.strip()
    if not candidate or candidate[0] not in "[{":
        return value
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return value


def _episode_to_json(result: EpisodeResult, trace_mode: str = "basic") -> str:
    full_trace = trace_mode == "full"
    payload = {
        "success": result.success,
        "metrics": _safe_data(result.metrics),
        "final_state": _compact_state_to_dict(result.final_state),
        "steps": [_full_step_to_dict(step) if full_trace else _compact_step_to_dict(step) for step in result.steps],
        "metadata": _safe_data(result.metadata),
    }
    if full_trace:
        payload["agentic_trace_mode"] = "full"
        payload["rollout"] = {
            "step_count": len(result.steps),
            "terminal": result.final_state.terminal,
            "success": result.success,
            "metrics": _safe_data(result.metrics),
        }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _compact_step_to_dict(step: EpisodeStep) -> dict[str, Any]:
    return {
        "step_idx": step.state.step_idx,
        "raw_model_output": step.raw_output.first_text() if step.raw_output is not None else None,
        "model_output": step.output.first_text() if step.output is not None else None,
        "action": _action_to_dict(step.parsed_action.action) if step.parsed_action is not None else None,
        "parse_error": step.parsed_action.error if step.parsed_action is not None else None,
        "reward": _safe_data(step.result.reward) if step.result is not None else None,
        "done": step.result.done if step.result is not None else None,
        "info": _safe_data(step.result.info) if step.result is not None else {},
    }


def _full_step_to_dict(step: EpisodeStep) -> dict[str, Any]:
    payload = _compact_step_to_dict(step)
    payload.update(
        {
            "state_before": _state_to_dict(step.state),
            "request": _agent_input_to_dict(step.request),
            "raw_output": _agent_output_to_dict(step.raw_output),
            "output": _agent_output_to_dict(step.output),
            "parsed_action": _parsed_action_to_dict(step.parsed_action),
            "result": _step_result_to_dict(step.result),
        }
    )
    return payload


def _compact_state_to_dict(state: EnvState) -> dict[str, Any]:
    return {
        "env_id": state.env_id,
        "step_idx": state.step_idx,
        "observation": _safe_data(state.observation),
        "terminal": state.terminal,
        "metadata": _safe_data(state.metadata),
    }


def _state_to_dict(state: EnvState) -> dict[str, Any]:
    payload = _compact_state_to_dict(state)
    payload["active_agent_ids"] = list(state.active_agent_ids)
    return payload


def _agent_input_to_dict(request: AgentInput | None) -> dict[str, Any] | None:
    if request is None:
        return None
    return {
        "first_text": request.first_text(),
        "content": [_content_block_to_dict(block) for block in request.content],
        "generation_kwargs": _safe_data(request.generation_kwargs),
        "metadata": _safe_data(request.metadata),
    }


def _agent_output_to_dict(output: AgentOutput | None) -> dict[str, Any] | None:
    if output is None:
        return None
    return {
        "first_text": output.first_text(),
        "content": [_content_block_to_dict(block) for block in output.content],
        "metadata": _safe_data(output.metadata),
    }


def _content_block_to_dict(block: ContentBlock) -> dict[str, Any]:
    return {
        "type": block.type,
        "data": _safe_data(block.data),
        "metadata": _safe_data(block.metadata),
    }


def _parsed_action_to_dict(parsed_action: ParsedAction | None) -> dict[str, Any] | None:
    if parsed_action is None:
        return None
    return {
        "action": _action_to_dict(parsed_action.action),
        "is_submit": parsed_action.is_submit,
        "error": parsed_action.error,
        "metadata": _safe_data(parsed_action.metadata),
    }


def _step_result_to_dict(result: StepResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "state_after": _state_to_dict(result.state),
        "reward": _safe_data(result.reward),
        "done": result.done,
        "info": _safe_data(result.info),
    }


def _action_to_dict(action: GameAction | dict[str, GameAction] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, dict):
        return {agent_id: _action_to_dict(agent_action) for agent_id, agent_action in action.items()}
    return {"type": action.type, "data": _safe_data(action.data), "agent_id": action.agent_id, "metadata": _safe_data(action.metadata)}


def _write_episode_artifacts(result: EpisodeResult, *, output_path: str | None, task_name: str, doc_id: int) -> dict[str, str]:
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

    return artifacts


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
        requested_action = _action_label(step.parsed_action.action if step.parsed_action is not None else None)
        rows.append(
            {
                "step_idx": step.state.step_idx,
                "model_output": step.output.first_text() if step.output is not None else None,
                "raw_model_output": step.raw_output.first_text() if step.raw_output is not None else None,
                "action": requested_action,
                "requested_action": requested_action,
                "executed_action": _executed_action_label(info, fallback=requested_action),
                "action_data": _safe_data(step.parsed_action.action.data) if step.parsed_action is not None and isinstance(step.parsed_action.action, GameAction) else None,
                "parse_error": step.parsed_action.error if step.parsed_action is not None else None,
                "env_error": _safe_data(info.get("error")),
                "invalid_actions": _safe_data(info.get("invalid_actions")),
                "reward": _safe_data(step.result.reward) if step.result is not None else None,
                "total_reward": _safe_data(info.get("total_reward")),
                "done": step.result.done if step.result is not None else None,
                "info": _safe_data(info),
            }
        )
    return rows


def _episode_summary_markdown(result: EpisodeResult, rows: list[dict[str, Any]]) -> str:
    metrics = result.metrics or {}
    requested_counts = Counter(row["requested_action"] for row in rows)
    executed_counts = Counter(row["executed_action"] for row in rows)
    lines = [
        "# Agentic Rollout Summary",
        "",
        f"- Success: {result.success}",
        f"- Steps: {len(rows)}",
        f"- Final terminal: {result.final_state.terminal}",
        f"- Metrics: `{json.dumps(_safe_data(metrics), ensure_ascii=False, sort_keys=True)}`",
        f"- Requested action counts: `{json.dumps(dict(requested_counts), ensure_ascii=False, sort_keys=True)}`",
        f"- Executed action counts: `{json.dumps(dict(executed_counts), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Steps",
        "",
        "| step | requested | executed | reward | total_reward | done | env_error | model_output |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {step_idx} | {requested} | {executed} | {reward} | {total_reward} | {done} | {env_error} | {model_output} |".format(
                step_idx=row["step_idx"],
                requested=_md_cell(row["requested_action"]),
                executed=_md_cell(row["executed_action"]),
                reward=_md_cell(row["reward"]),
                total_reward=_md_cell(row["total_reward"]),
                done=_md_cell(row["done"]),
                env_error=_md_cell(row["env_error"]),
                model_output=_md_cell(row["model_output"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _episode_video_frames(result: EpisodeResult) -> list[Any]:
    frames = []
    for step in result.steps:
        frame = _state_screen_frame(step.state)
        if frame is not None:
            frames.append(frame)
    final_frame = _state_screen_frame(result.final_state)
    if final_frame is not None:
        frames.append(final_frame)
    return frames


def _state_screen_frame(state: EnvState) -> Any:
    observation = state.observation if isinstance(state.observation, dict) else {}
    for key in ("screen_buffer", "frame", "image"):
        frame = observation.get(key)
        if frame is not None:
            return frame
    return None


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

    if _looks_like_pil_image(frame):
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


def _action_label(action: GameAction | dict[str, GameAction] | None) -> str:
    if action is None:
        return "NONE"
    if isinstance(action, dict):
        return ",".join(f"{agent_id}:{_action_label(agent_action)}" for agent_id, agent_action in action.items())
    data = action.data if isinstance(action.data, dict) else {}
    buttons = data.get("buttons")
    if isinstance(buttons, dict):
        active = [name for name, value in buttons.items() if value]
        return "+".join(active) if active else "NOOP"
    if isinstance(buttons, list):
        return "+".join(str(item) for item in buttons) if buttons else "NOOP"
    return action.type


def _executed_action_label(info: dict[str, Any], *, fallback: str) -> str:
    buttons = info.get("buttons")
    if isinstance(buttons, dict):
        active = [name for name, value in buttons.items() if value]
        return "+".join(active) if active else "NOOP"
    return fallback


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _safe_data(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "length": len(value)}

    scalar = _numpy_scalar_to_python(value)
    if scalar is not value:
        return scalar

    if _looks_like_array_or_tensor(value):
        return {
            "type": _type_name(value),
            "shape": _safe_shape(getattr(value, "shape", None)),
            "dtype": str(getattr(value, "dtype", "")),
        }
    if _looks_like_pil_image(value):
        return {
            "type": _type_name(value),
            "size": _safe_data(getattr(value, "size", None), depth=depth + 1, seen=seen),
            "mode": getattr(value, "mode", None),
            "format": getattr(value, "format", None),
        }

    if depth >= 8:
        return {"type": _type_name(value), "repr": _safe_repr(value)}

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return {"type": _type_name(value), "repr": "<recursive>"}
    seen.add(value_id)
    try:
        if isinstance(value, dict):
            return {_safe_key(key): _safe_data(item, depth=depth + 1, seen=seen) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe_data(item, depth=depth + 1, seen=seen) for item in value]
        if isinstance(value, set):
            return [_safe_data(item, depth=depth + 1, seen=seen) for item in sorted(value, key=repr)]
    finally:
        seen.discard(value_id)

    return {"type": _type_name(value), "repr": _safe_repr(value)}


def _safe_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(_safe_data(value))


def _numpy_scalar_to_python(value: Any) -> Any:
    module = getattr(value.__class__, "__module__", "")
    if not module.startswith("numpy"):
        return value
    item = getattr(value, "item", None)
    if not callable(item):
        return value
    try:
        scalar = item()
    except Exception:
        return value
    if scalar is None or isinstance(scalar, str | int | float | bool):
        return scalar
    return value


def _looks_like_array_or_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _looks_like_pil_image(value: Any) -> bool:
    return getattr(value.__class__, "__module__", "").startswith("PIL.") and hasattr(value, "size") and hasattr(value, "mode")


def _safe_shape(shape: Any) -> Any:
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except TypeError:
        return str(shape)


def _type_name(value: Any) -> str:
    cls = value.__class__
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__qualname__", cls.__name__)
    return name if module in {"", "builtins"} else f"{module}.{name}"


def _safe_repr(value: Any, max_chars: int = 500) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {_type_name(value)}>"
    if len(text) > max_chars:
        return f"{text[:max_chars]}...<truncated>"
    return text

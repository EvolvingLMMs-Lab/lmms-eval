"""lmms-eval glue: `generate_until_game` Instances -> episode rollouts -> responses.

Component ownership is split once, in one place:

- task YAML owns the environment side: ``game_env``, ``observation_parser``,
  ``action_parser`` (per-instance, from ``Instance.args``); the parsers may be
  omitted, in which case the loop uses ``TemplateObservationParser`` and an
  action parser built from ``env.action_spec()``;
- the CLI owns the model side: ``--agentic_model_server(_args)`` and
  ``--agentic_output_parser(_args)`` (one shared server per run);
- loop options ride in ``generation_kwargs``: ``max_game_steps``,
  ``game_seed``, ``multiturn``, ``history_turns``.

A raising episode never kills the run: after ``--agentic_episode_retries``
extra attempts it is recorded as a failed ``EpisodeResult`` with metric
``env_error=1.0`` and the error in ``metadata``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.agentic.artifacts import write_episode_artifacts
from lmms_eval.agentic.components import resolve
from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.episode import run_episode
from lmms_eval.agentic.parsers import (
    ActionParser,
    ModelOutputParser,
    ObservationParser,
    TemplateObservationParser,
    build_action_parser,
)
from lmms_eval.agentic.servers import ModelServer
from lmms_eval.agentic.trace import episode_to_json
from lmms_eval.agentic.types import EnvState, EpisodeResult
from lmms_eval.api.instance import Instance
from lmms_eval.utils import simple_parse_args_string


def run_generate_until_game(lm: Any, requests: list[Instance], response_cache: Any = None, cli_args: Any = None) -> list[str]:
    del response_cache  # rollouts are env-coupled and not response-cacheable
    output_path = getattr(cli_args, "output_path", None) if cli_args is not None else None
    max_parallel = max(1, int(getattr(cli_args, "agentic_max_parallel_rollouts", None) or 1))
    episode_retries = max(0, int(getattr(cli_args, "agentic_episode_retries", None) or 0))
    model_server = resolve("model_server", _cli_spec(cli_args, "agentic_model_server", default="openai"), expected=ModelServer)
    model_output_parser = resolve("model_output_parser", _cli_spec(cli_args, "agentic_output_parser", default="identity"), expected=ModelOutputParser)

    plans = [_EpisodePlan.from_instance(req, lm) for req in requests]

    def build_components(plan: "_EpisodePlan") -> tuple[EnvManager, ObservationParser, ActionParser]:
        """Resolve per-episode components. Errors here are task wiring bugs and fail the run loudly."""

        context = {"doc": plan.doc, "lmms_eval_specific_kwargs": plan.lmms_eval_specific_kwargs}
        env = resolve("env_manager", plan.game_env_spec, expected=EnvManager, **context)
        try:
            if plan.observation_parser_spec is None:
                observation_parser: ObservationParser = TemplateObservationParser()
            else:
                observation_parser = resolve("observation_parser", plan.observation_parser_spec, expected=ObservationParser, **context)
            if plan.action_parser_spec is None:
                action_parser: ActionParser = build_action_parser(env.action_spec())
            else:
                action_parser = resolve("action_parser", plan.action_parser_spec, expected=ActionParser, **context)
        except BaseException:
            env.close()
            raise
        return env, observation_parser, action_parser

    def run_one(plan: "_EpisodePlan") -> str:
        attempts = episode_retries + 1
        for attempt in range(1, attempts + 1):
            env, observation_parser, action_parser = build_components(plan)
            try:
                episode = run_episode(
                    env=env,
                    observation_parser=observation_parser,
                    model_output_parser=model_output_parser,
                    action_parser=action_parser,
                    model_server=model_server,
                    doc=plan.doc,
                    max_steps=plan.max_steps,
                    seed=plan.seed,
                    multiturn=plan.multiturn,
                    history_turns=plan.history_turns,
                    generation_kwargs=plan.generation_kwargs,
                    request_metadata={"lmms_eval": {"doc_id": plan.doc_id, "task_name": plan.task_name, "split": plan.split}},
                )
                artifacts = write_episode_artifacts(episode, output_path=output_path, task_name=plan.task_name, doc_id=plan.doc_id)
                if artifacts:
                    episode.metadata = {**episode.metadata, "artifacts": artifacts}
                return episode_to_json(episode)
            except Exception as exc:  # noqa: BLE001 - envs and backends raise arbitrary types
                error = exc
                eval_logger.warning(f"[agentic] episode failed (task={plan.task_name} doc={plan.doc_id} attempt {attempt}/{attempts}): {exc!r}")
        return episode_to_json(_error_episode(plan, error, attempts))

    progress = tqdm(total=len(plans), desc="Agentic rollouts", disable=not plans)
    try:
        if max_parallel <= 1:
            results = []
            for plan in plans:
                results.append(run_one(plan))
                progress.update(1)
            return results
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(plans))) as executor:
            futures = [executor.submit(run_one, plan) for plan in plans]
            results = []
            for future in futures:
                results.append(future.result())
                progress.update(1)
            return results
    finally:
        progress.close()


@dataclass(slots=True)
class _EpisodePlan:
    doc: Any
    generation_kwargs: dict[str, Any]
    max_steps: int
    seed: int | None
    multiturn: bool
    history_turns: int | None
    game_env_spec: Any
    observation_parser_spec: Any
    action_parser_spec: Any
    lmms_eval_specific_kwargs: dict[str, Any] | None
    doc_id: int
    task_name: str
    split: str

    @classmethod
    def from_instance(cls, req: Instance, lm: Any) -> "_EpisodePlan":
        if len(req.args) != 10:
            raise ValueError(f"generate_until_game expects 10-element Instance.args (see ConfigurableTask.construct_requests), got {len(req.args)}")
        _ctx, generation_kwargs, _doc_to_visual, game_env, observation_parser, action_parser, lmms_eval_specific_kwargs, doc_id, task_name, split = req.args

        gen_kwargs = dict(generation_kwargs or {})
        max_steps = int(gen_kwargs.pop("max_game_steps", gen_kwargs.pop("max_agentic_steps", 32)))
        seed = gen_kwargs.pop("game_seed", None)
        multiturn = _as_bool(gen_kwargs.pop("multiturn", False))
        history_turns = _as_history_turns(gen_kwargs.pop("history_turns", 6))
        return cls(
            doc=lm.task_dict[task_name][split][doc_id],
            generation_kwargs=gen_kwargs,
            max_steps=max_steps,
            seed=seed,
            multiturn=multiturn,
            history_turns=history_turns,
            game_env_spec=game_env,
            observation_parser_spec=observation_parser,
            action_parser_spec=action_parser,
            lmms_eval_specific_kwargs=lmms_eval_specific_kwargs,
            doc_id=int(doc_id),
            task_name=str(task_name),
            split=str(split),
        )


def _error_episode(plan: "_EpisodePlan", error: Exception, attempts: int) -> EpisodeResult:
    """Failed rollout stand-in so one broken episode never kills the run."""

    final_state = EnvState(env_id=plan.task_name, step_idx=0, observation=None, terminal=True, metadata={"error": repr(error)})
    return EpisodeResult(
        final_state=final_state,
        steps=[],
        success=False,
        metrics={"env_error": 1.0},
        metadata={"error": repr(error), "error_type": type(error).__name__, "attempts": attempts, "max_steps": plan.max_steps},
    )


def _cli_spec(cli_args: Any, name_attr: str, *, default: str) -> Any:
    """Combine ``--<name_attr>`` and ``--<name_attr>_args`` into one spec."""

    name = getattr(cli_args, name_attr, None) if cli_args is not None else None
    args = getattr(cli_args, f"{name_attr}_args", "") if cli_args is not None else ""
    kwargs = simple_parse_args_string(args) if isinstance(args, str) and args else dict(args or {})
    kwargs = {key: _decode_json_value(value) for key, value in kwargs.items()}
    if not kwargs:
        return name or default
    return {"name": name or default, **kwargs}


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_history_turns(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "all", "none", "null"}:
            return None
        value = int(normalized)
    return max(0, int(value))

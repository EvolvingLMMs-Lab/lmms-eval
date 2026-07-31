"""The rollout loop: one episode, one plain function.

The runner parallelizes episodes with a thread pool, so this loop stays
synchronous and single-agent: observe -> generate -> parse -> step, until the
environment is terminal or ``max_steps`` is reached.
"""

from __future__ import annotations

from typing import Any

from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.parsers import (
    ActionParser,
    ModelOutputParser,
    ObservationParser,
    ParserContext,
)
from lmms_eval.agentic.servers import ModelServer
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    EnvState,
    EpisodeResult,
    EpisodeStep,
    GameAction,
)


def run_episode(
    *,
    env: EnvManager,
    observation_parser: ObservationParser,
    model_output_parser: ModelOutputParser,
    action_parser: ActionParser,
    model_server: ModelServer,
    doc: Any,
    max_steps: int = 32,
    seed: int | None = None,
    multiturn: bool = False,
    history_turns: int | None = 6,
    generation_kwargs: dict[str, Any] | None = None,
    request_metadata: dict[str, Any] | None = None,
    agent_id: str = "agent",
) -> EpisodeResult:
    generation_kwargs = dict(generation_kwargs or {})
    request_metadata = dict(request_metadata or {})
    steps: list[EpisodeStep] = []
    history: list[dict[str, Any]] = []
    action_spec = env.action_spec()

    try:
        state = env.reset(doc, seed=seed)
        while not state.terminal and len(steps) < max_steps:
            ctx = ParserContext(state=state, agent_id=agent_id, step_idx=state.step_idx, history=list(history), metadata={"max_steps": max_steps, "doc": doc, "action_spec": action_spec})
            request = observation_parser.parse(state, ctx)
            if not isinstance(request, AgentInput):
                raise TypeError(f"observation parser must return AgentInput, got {type(request).__name__}")
            request.generation_kwargs = {**generation_kwargs, **request.generation_kwargs}
            request.metadata = {**request_metadata, **request.metadata}
            if multiturn:
                visible_history = _history_window(history, history_turns)
                if visible_history:
                    request.metadata["conversation_history"] = visible_history
                    request.metadata["conversation_history_turns"] = len(visible_history) // 2

            raw_output = model_server.generate(request)
            ctx.request = request
            ctx.raw_output = raw_output
            output = model_output_parser.parse(raw_output, ctx)
            parsed = action_parser.parse(output, ctx)
            action = parsed.action if parsed.action is not None else GameAction(type="parse_error", data=parsed.error, agent_id=agent_id)
            result = env.step(action)

            steps.append(EpisodeStep(state=state, request=request, raw_output=raw_output, output=output, parsed_action=parsed, result=result))
            if multiturn:
                history.extend(_history_turns_for(request, raw_output, state=state, agent_id=agent_id))
            state = result.state

        return EpisodeResult(
            final_state=state,
            steps=steps,
            success=_state_success(state),
            metrics=_state_metrics(state),
            metadata={"max_steps": max_steps, "agent_id": agent_id, "multiturn": multiturn, "history_turns": history_turns},
        )
    finally:
        env.close()


def _state_metrics(state: EnvState) -> dict[str, float]:
    metrics = state.metadata.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _state_success(state: EnvState) -> bool | None:
    success = state.metadata.get("success")
    return success if isinstance(success, bool) else None


def _history_window(history: list[dict[str, Any]], history_turns: int | None) -> list[dict[str, Any]]:
    if history_turns is None:
        return list(history)
    if history_turns <= 0:
        return []
    return list(history[-2 * history_turns :])


def _history_turns_for(request: AgentInput, raw_output: AgentOutput, *, state: EnvState, agent_id: str) -> list[dict[str, Any]]:
    assistant_text = raw_output.first_text() if isinstance(raw_output, AgentOutput) else None
    turn_metadata = {"step_idx": state.step_idx, "agent_id": agent_id}
    return [
        {"role": "user", "content": list(request.content), "metadata": turn_metadata},
        {"role": "assistant", "content": assistant_text or "", "metadata": dict(turn_metadata)},
    ]

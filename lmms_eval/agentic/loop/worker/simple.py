from __future__ import annotations

from typing import Any

from lmms_eval.agentic.env import GameEnv
from lmms_eval.agentic.loop.session import LoopSession
from lmms_eval.agentic.loop.worker.base import LoopWorker
from lmms_eval.agentic.model_server import ModelServer
from lmms_eval.agentic.parsers import (
    ActionParser,
    ModelOutputParser,
    ObservationParser,
    ParserContext,
)
from lmms_eval.agentic.types import (
    AgentInput,
    EnvState,
    EpisodeResult,
    EpisodeStep,
    GameAction,
)


class SimpleLoopWorker(LoopWorker):
    """Default single-agent loop."""

    def __init__(
        self,
        model_server: ModelServer,
        env: GameEnv,
        observation_parser: ObservationParser,
        action_parser: ActionParser,
        max_steps: int = 32,
        model_output_parser: ModelOutputParser | None = None,
        multiturn: bool | str = False,
        history_turns: int | str | None = 6,
        generation_kwargs: dict[str, Any] | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_server = model_server
        self.env = env
        self.observation_parser = observation_parser
        self.model_output_parser = model_output_parser
        self.action_parser = action_parser
        self.max_steps = max_steps
        self.multiturn = _as_bool(multiturn)
        self.history_turns = _normalize_history_turns(history_turns)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.request_metadata = dict(request_metadata or {})

    def run(self, doc: Any, seed: int | None = None, agent_id: str = "agent") -> EpisodeResult:
        session = self.new_session(doc, seed=seed, agent_id=agent_id)
        while not session.done:
            request = session.next_request()
            if request is None:
                break
            session.apply_model_output(self.model_server.generate(request))
        return session.result()

    def new_session(self, doc: Any, seed: int | None = None, agent_id: str = "agent") -> "SimpleLoopSession":
        return SimpleLoopSession(
            env=self.env,
            observation_parser=self.observation_parser,
            action_parser=self.action_parser,
            max_steps=self.max_steps,
            model_output_parser=self.model_output_parser,
            multiturn=self.multiturn,
            history_turns=self.history_turns,
            generation_kwargs=self.generation_kwargs,
            request_metadata=self.request_metadata,
            doc=doc,
            seed=seed,
            agent_id=agent_id,
        )


class SimpleLoopSession(LoopSession):
    def __init__(
        self,
        *,
        env: GameEnv,
        observation_parser: ObservationParser,
        action_parser: ActionParser,
        max_steps: int,
        model_output_parser: ModelOutputParser | None,
        multiturn: bool,
        history_turns: int | None,
        generation_kwargs: dict[str, Any],
        request_metadata: dict[str, Any],
        doc: Any,
        seed: int | None,
        agent_id: str,
    ) -> None:
        self.env = env
        self.observation_parser = observation_parser
        self.action_parser = action_parser
        self.max_steps = max_steps
        self.model_output_parser = model_output_parser
        self.multiturn = multiturn
        self.history_turns = history_turns
        self.generation_kwargs = dict(generation_kwargs or {})
        self.request_metadata = dict(request_metadata or {})
        self.agent_id = agent_id

        self.state = self.env.reset(doc, seed=seed)
        self.steps: list[EpisodeStep] = []
        self.history: list[dict[str, Any]] = []
        self._pending_state: EnvState | None = None
        self._pending_request: Any | None = None

    @property
    def done(self) -> bool:
        if self._pending_request is not None:
            return False
        return self.state.terminal or len(self.steps) >= self.max_steps

    def next_request(self) -> Any | None:
        if self.done:
            return None
        if self._pending_request is not None:
            return self._pending_request

        ctx = self._parser_context(state=self.state)
        request = self.observation_parser.parse(self.state, ctx)
        if self.multiturn:
            visible_history = _history_window(self.history, self.history_turns)
            if visible_history and isinstance(request, AgentInput):
                request.metadata = {
                    **request.metadata,
                    "conversation_history": visible_history,
                    "conversation_history_turns": len(visible_history) // 2,
                }
        if isinstance(request, AgentInput):
            request.generation_kwargs = {**self.generation_kwargs, **(request.generation_kwargs or {})}
            request.metadata = {**self.request_metadata, **(request.metadata or {})}
        self._pending_state = self.state
        self._pending_request = request
        return request

    def apply_model_output(self, raw_output: Any) -> None:
        if self._pending_state is None or self._pending_request is None:
            raise RuntimeError("SimpleLoopSession.apply_model_output called without a pending request")

        state = self._pending_state
        request = self._pending_request
        output_ctx = self._parser_context(state=state, request=request, raw_output=raw_output)
        output = self.model_output_parser.parse(raw_output, output_ctx) if self.model_output_parser is not None else raw_output
        action_ctx = self._parser_context(state=state, request=request, raw_output=raw_output)
        parsed = self.action_parser.parse(output, action_ctx)
        action = parsed.action if parsed.action is not None else GameAction(type="parse_error", data=parsed.error, agent_id=self.agent_id)
        result = self.env.step(action)
        self.steps.append(EpisodeStep(state=state, request=request, raw_output=raw_output, output=output, parsed_action=parsed, result=result))
        if self.multiturn:
            self.history.extend(_history_turns(request, raw_output, state=state, agent_id=self.agent_id))
        self.state = result.state
        self._pending_state = None
        self._pending_request = None

    def result(self) -> EpisodeResult:
        metrics = self.state.metadata.get("metrics", {}) if isinstance(self.state.metadata.get("metrics"), dict) else {}
        success = self.state.metadata.get("success")
        return EpisodeResult(
            final_state=self.state,
            steps=self.steps,
            success=success if isinstance(success, bool) else None,
            metrics=metrics,
            metadata={"max_steps": self.max_steps, "agent_id": self.agent_id, "multiturn": self.multiturn, "history_turns": self.history_turns},
        )

    def _parser_context(self, *, state: EnvState, request: Any = None, raw_output: Any = None) -> ParserContext:
        return ParserContext(
            state=state,
            agent_id=self.agent_id,
            step_idx=state.step_idx,
            request=request,
            raw_output=raw_output,
            history=list(self.history),
            metadata={"max_steps": self.max_steps},
        )


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_history_turns(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "all", "none", "null"}:
            return None
        value = int(normalized)
    return max(0, int(value))


def _history_window(history: list[dict[str, Any]], history_turns: int | None) -> list[dict[str, Any]]:
    if not history:
        return []
    if history_turns is None:
        return list(history)
    if history_turns <= 0:
        return []
    return list(history[-2 * history_turns :])


def _history_turns(request: Any, raw_output: Any, *, state: EnvState, agent_id: str) -> list[dict[str, Any]]:
    if not isinstance(request, AgentInput):
        return []
    assistant_content = raw_output.first_text() if hasattr(raw_output, "first_text") else None
    return [
        {
            "role": "user",
            "content": list(request.content),
            "metadata": {"step_idx": state.step_idx, "agent_id": agent_id},
        },
        {
            "role": "assistant",
            "content": assistant_content or "",
            "metadata": {"step_idx": state.step_idx, "agent_id": agent_id},
        },
    ]

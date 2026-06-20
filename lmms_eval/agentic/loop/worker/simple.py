from __future__ import annotations

from typing import Any

from lmms_eval.agentic.env import GameEnv
from lmms_eval.agentic.loop.worker.base import LoopWorker
from lmms_eval.agentic.model_server import ModelServer
from lmms_eval.agentic.parsers import ActionParser, ModelOutputParser, ObservationParser
from lmms_eval.agentic.types import EpisodeResult, EpisodeStep, GameAction


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
    ) -> None:
        self.model_server = model_server
        self.env = env
        self.observation_parser = observation_parser
        self.model_output_parser = model_output_parser
        self.action_parser = action_parser
        self.max_steps = max_steps

    def run(self, doc: Any, seed: int | None = None, agent_id: str = "agent") -> EpisodeResult:
        state = self.env.reset(doc, seed=seed)
        steps: list[EpisodeStep] = []
        for _ in range(self.max_steps):
            if state.terminal:
                break
            request = self.observation_parser.parse(state, agent_id=agent_id)
            raw_output = self.model_server.generate(request)
            output = self.model_output_parser.parse(raw_output, state, agent_id=agent_id) if self.model_output_parser is not None else raw_output
            parsed = self.action_parser.parse(output, state, agent_id=agent_id)
            action = parsed.action if parsed.action is not None else GameAction(type="parse_error", data=parsed.error, agent_id=agent_id)
            result = self.env.step(action)
            steps.append(EpisodeStep(state=state, request=request, raw_output=raw_output, output=output, parsed_action=parsed, result=result))
            state = result.state
            if result.done:
                break

        metrics = state.metadata.get("metrics", {}) if isinstance(state.metadata.get("metrics"), dict) else {}
        success = state.metadata.get("success")
        return EpisodeResult(final_state=state, steps=steps, success=success if isinstance(success, bool) else None, metrics=metrics, metadata={"max_steps": self.max_steps, "agent_id": agent_id})

from __future__ import annotations

from lmms_eval.agentic.loop.worker.simple import SimpleLoopWorker
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    GameAction,
    ParsedAction,
    StepResult,
)


class _Env:
    def reset(self, doc, seed=None):
        return EnvState(env_id="test", step_idx=0, observation={"doc": doc, "seed": seed})

    def step(self, action):
        step_idx = int(action.data["step_idx"]) + 1
        return StepResult(state=EnvState(env_id="test", step_idx=step_idx, observation={}, terminal=step_idx >= 2), reward=0.0, done=step_idx >= 2)


class _ObservationParser:
    def parse(self, state, agent_id=None):
        return AgentInput(content=[ContentBlock.text(f"obs {state.step_idx}")], metadata={"step_idx": state.step_idx, "agent_id": agent_id})


class _ModelServer:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return AgentOutput(content=[ContentBlock.text(f"act {request.metadata['step_idx']}")])


class _ActionParser:
    def parse(self, output, state, agent_id=None):
        return ParsedAction(action=GameAction(type="test_action", data={"step_idx": state.step_idx}, agent_id=agent_id))


def test_simple_loop_worker_can_attach_multiturn_history():
    model_server = _ModelServer()
    worker = SimpleLoopWorker(
        model_server=model_server,
        env=_Env(),
        observation_parser=_ObservationParser(),
        action_parser=_ActionParser(),
        max_steps=2,
        multiturn=True,
        history_turns=1,
    )

    result = worker.run({"id": "doc"}, seed=7)

    assert len(result.steps) == 2
    assert "conversation_history" not in model_server.requests[0].metadata
    history = model_server.requests[1].metadata["conversation_history"]
    assert [turn["role"] for turn in history] == ["user", "assistant"]
    assert history[0]["content"][0].data == "obs 0"
    assert history[1]["content"] == "act 0"
    assert model_server.requests[1].metadata["conversation_history_turns"] == 1

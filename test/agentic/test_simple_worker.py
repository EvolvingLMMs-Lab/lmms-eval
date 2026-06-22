from __future__ import annotations

from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.loop.worker.simple import SimpleLoopWorker
from lmms_eval.agentic.registry import build_env_manager
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    GameAction,
    ParsedAction,
    StepResult,
)


class _Env(EnvManager):
    def __init__(self):
        self.state = None

    def reset(self, doc, seed=None):
        self.state = EnvState(env_id="test", step_idx=0, observation={"doc": doc, "seed": seed})
        return self.state

    def step(self, action):
        step_idx = int(action.data["step_idx"]) + 1
        self.state = EnvState(env_id="test", step_idx=step_idx, observation={}, terminal=step_idx >= 2)
        return StepResult(state=self.state, reward=0.0, done=step_idx >= 2)

    def get_state(self):
        return self.state


class _ObservationParser:
    def parse(self, state, ctx):
        agent_id = ctx.agent_id
        return AgentInput(content=[ContentBlock.text(f"obs {state.step_idx}")], metadata={"step_idx": state.step_idx, "agent_id": agent_id})


class _ModelServer:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return AgentOutput(content=[ContentBlock.text(f"act {request.metadata['step_idx']}")])


class _ActionParser:
    def parse(self, output, ctx):
        state = ctx.state
        agent_id = ctx.agent_id
        return ParsedAction(action=GameAction(type="test_action", data={"step_idx": state.step_idx}, agent_id=agent_id))


def test_env_manager_tracks_state():
    manager = _Env()

    state = manager.reset({"id": "doc"}, seed=7)
    result = manager.step(GameAction(type="test_action", data={"step_idx": state.step_idx}))

    assert state.env_id == "test"
    assert result.state.step_idx == 1
    assert manager.get_state().step_idx == 1


def test_build_env_manager_builds_factory():
    manager = build_env_manager(lambda: _Env())

    state = manager.reset("doc")

    assert isinstance(manager, EnvManager)
    assert state.env_id == "test"


def test_simple_loop_worker_can_attach_multiturn_history():
    model_server = _ModelServer()
    worker = SimpleLoopWorker(
        model_server=model_server,
        env_manager=_Env(),
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


def test_simple_loop_worker_can_reuse_single_env_manager_after_result():
    worker = SimpleLoopWorker(
        model_server=_ModelServer(),
        env_manager=_Env(),
        observation_parser=_ObservationParser(),
        action_parser=_ActionParser(),
        max_steps=1,
    )

    first = worker.run({"id": "first"})
    second = worker.run({"id": "second"})

    assert first.final_state.env_id == "test"
    assert second.final_state.env_id == "test"


class _AnyPayloadEnv(EnvManager):
    def __init__(self):
        self.state = None

    def reset(self, doc, seed=None):
        self.state = EnvState(env_id="any", step_idx=0, observation={"doc": doc, "seed": seed})
        return self.state

    def step(self, action):
        self.state = EnvState(env_id="any", step_idx=1, observation={}, terminal=True)
        return StepResult(state=self.state, reward=1.0, done=True, info={"action_data": action.data})

    def get_state(self):
        return self.state


class _AnyObservationParser:
    def parse(self, state, ctx):
        return ("tensor_like_request", state.observation["doc"], ctx.agent_id)


class _AnyModelServer:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return {"policy_logits": [0.1, 0.9], "request": request}


class _AnyActionParser:
    def parse(self, output, ctx):
        return ParsedAction(action=GameAction(type="policy_action", data=output, agent_id=ctx.agent_id))


def test_simple_loop_worker_accepts_arbitrary_parser_payloads():
    model_server = _AnyModelServer()
    worker = SimpleLoopWorker(
        model_server=model_server,
        env_manager=_AnyPayloadEnv(),
        observation_parser=_AnyObservationParser(),
        action_parser=_AnyActionParser(),
        max_steps=1,
    )

    result = worker.run("doc", seed=3)

    assert model_server.requests == [("tensor_like_request", "doc", "agent")]
    assert result.steps[0].request == ("tensor_like_request", "doc", "agent")
    assert result.steps[0].raw_output == {"policy_logits": [0.1, 0.9], "request": ("tensor_like_request", "doc", "agent")}
    assert result.steps[0].parsed_action.action.data["policy_logits"] == [0.1, 0.9]

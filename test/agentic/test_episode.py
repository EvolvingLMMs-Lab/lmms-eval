from __future__ import annotations

import pytest

from lmms_eval.agentic import (
    FixedActionModelServer,
    IdentityModelOutputParser,
    run_episode,
)
from lmms_eval.agentic.parsers import ObservationParser, ParserContext

from .conftest import ScriptedEnv, TextObservationParser, UppercaseActionParser


def _run(env, *, model_server=None, **kwargs):
    return run_episode(
        env=env,
        observation_parser=kwargs.pop("observation_parser", TextObservationParser()),
        model_output_parser=IdentityModelOutputParser(),
        action_parser=UppercaseActionParser(),
        model_server=model_server or FixedActionModelServer(action="attack"),
        doc={"instruction": "win"},
        **kwargs,
    )


def test_episode_runs_to_terminal_state():
    env = ScriptedEnv(episode_len=3)
    result = _run(env)

    assert len(result.steps) == 3
    assert result.final_state.terminal is True
    assert result.success is True
    assert result.metrics == {"scripted_steps": 3.0}
    assert [action.type for action in env.actions] == ["ATTACK", "ATTACK", "ATTACK"]
    assert env.closed is True


def test_episode_respects_max_steps():
    env = ScriptedEnv(episode_len=100)
    result = _run(env, max_steps=5)

    assert len(result.steps) == 5
    assert result.final_state.terminal is False
    assert result.metadata["max_steps"] == 5


def test_episode_merges_generation_kwargs_and_metadata():
    env = ScriptedEnv(episode_len=1)
    result = _run(env, generation_kwargs={"temperature": 0.5}, request_metadata={"lmms_eval": {"doc_id": 7}})

    request = result.steps[0].request
    assert request.generation_kwargs["temperature"] == 0.5
    assert request.metadata["lmms_eval"] == {"doc_id": 7}


def test_episode_forwards_seed_to_env_reset():
    env = ScriptedEnv(episode_len=1)
    _run(env, seed=1234)

    assert env.reset_seed == 1234


def test_episode_multiturn_attaches_conversation_history():
    env = ScriptedEnv(episode_len=3)
    result = _run(env, multiturn=True, history_turns=1)

    first, second, third = result.steps
    assert "conversation_history" not in first.request.metadata
    assert second.request.metadata["conversation_history_turns"] == 1
    # history_turns=1 keeps only the latest user/assistant pair
    assert len(third.request.metadata["conversation_history"]) == 2
    assert third.request.metadata["conversation_history"][1]["content"] == "attack"


def test_episode_records_parse_error_as_parse_error_action():
    env = ScriptedEnv(episode_len=1)
    result = _run(env, model_server=FixedActionModelServer(action=""))

    step = result.steps[0]
    assert step.parsed_action.error == "empty output"
    assert env.actions[0].type == "parse_error"


def test_episode_closes_env_when_a_component_raises():
    class ExplodingParser(ObservationParser):
        def parse(self, state, ctx: ParserContext):
            raise RuntimeError("boom")

    env = ScriptedEnv(episode_len=3)
    with pytest.raises(RuntimeError, match="boom"):
        _run(env, observation_parser=ExplodingParser())

    assert env.closed is True


def test_episode_rejects_non_agent_input_observation():
    class WrongTypeParser(ObservationParser):
        def parse(self, state, ctx: ParserContext):
            return "not an AgentInput"

    env = ScriptedEnv(episode_len=3)
    with pytest.raises(TypeError, match="AgentInput"):
        _run(env, observation_parser=WrongTypeParser())

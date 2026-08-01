from __future__ import annotations

import pytest

from lmms_eval.agentic import (
    FixedActionModelServer,
    ParserContext,
    run_episode,
)

from .conftest import ScriptedEnv, text_observation_parser, uppercase_action_parser


def _run(env, *, model_server=None, **kwargs):
    return run_episode(
        env=env,
        observation_pipeline=kwargs.pop("observation_pipeline", text_observation_parser),
        action_pipeline=kwargs.pop("action_pipeline", uppercase_action_parser),
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
    def exploding_parser(value, context: ParserContext):
        del value, context
        raise RuntimeError("boom")

    env = ScriptedEnv(episode_len=3)
    with pytest.raises(RuntimeError, match="boom"):
        _run(env, observation_pipeline=exploding_parser)

    assert env.closed is True


def test_episode_rejects_non_agent_input_observation():
    def wrong_type_parser(value, context: ParserContext):
        del value, context
        return "not an AgentInput"

    env = ScriptedEnv(episode_len=3)
    with pytest.raises(TypeError, match="AgentInput"):
        _run(env, observation_pipeline=wrong_type_parser)


def test_episode_runs_any_to_any_action_pipeline_in_order():
    def unwrap_text(value, context):
        del context
        return value.first_text()

    def text_to_action(value, context):
        from lmms_eval.agentic import GameAction, ParsedAction

        return ParsedAction(action=GameAction(type=value.upper(), agent_id=context.agent_id))

    env = ScriptedEnv(episode_len=1)
    result = _run(env, action_pipeline=[unwrap_text, text_to_action], model_name="Qwen/Qwen3.6-27B")

    assert result.steps[0].parsed_action.action.type == "ATTACK"

"""VizDoom task components exercised without the vizdoom binary."""

from __future__ import annotations

import json

from lmms_eval.agentic import AgentInput, AgentOutput, ContentBlock, EnvState
from lmms_eval.agentic.parsers import ParserContext
from lmms_eval.tasks.vizdoom_agentic import utils as vizdoom_utils


def _ctx(state, agent_id="agent"):
    return ParserContext(state=state, agent_id=agent_id, step_idx=state.step_idx)


def _state(observation=None):
    return EnvState(env_id="vizdoom", step_idx=0, observation=observation or {})


def test_action_parser_reads_plain_button_combo():
    parser = vizdoom_utils._sibling_module("parsers").VizDoomActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("MOVE_LEFT + ATTACK")]), _ctx(_state()))

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": ["MOVE_LEFT", "ATTACK"]}


def test_action_parser_reads_json_button_values():
    parser = vizdoom_utils._sibling_module("parsers").VizDoomActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text('{"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}')]), _ctx(_state()))

    assert parsed.action.data == {"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}


def test_action_parser_errors_on_gibberish():
    parser = vizdoom_utils._sibling_module("parsers").VizDoomActionParser(buttons=["ATTACK"])
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("no button here")]), _ctx(_state()))

    assert parsed.action is None
    assert parsed.error is not None


def test_observation_parser_human_view_emits_prompt_only_text():
    parser = vizdoom_utils.vizdoom_observation_parser()
    state = _state({"screen_buffer": None, "game_variables": {"HEALTH": 100}})

    request = parser.parse(state, _ctx(state))

    assert isinstance(request, AgentInput)
    text = request.first_text()
    assert text
    # human-view mode must not leak oracle game variables into the prompt
    assert "HEALTH" not in text


def test_env_manager_factory_builds_without_vizdoom_installed():
    manager = vizdoom_utils.vizdoom_env_manager()

    assert manager.config["screen_resolution"] == "RES_320X240"
    assert manager.config["available_buttons"] == ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
    assert manager.frame_history == 5
    assert manager.tics_per_action == 5


def test_process_results_reads_episode_metrics_with_embedded_qwen_thinking():
    payload = {
        "success": False,
        "metrics": {
            "vizdoom_success": 0.0,
            "vizdoom_steps": 25.0,
            "vizdoom_invalid_actions": 1.0,
        },
        "steps": [{"raw_model_output": "aim, then fire</think>"}],
    }

    metrics = vizdoom_utils.vizdoom_process_results({}, [json.dumps(payload)])

    assert metrics == {
        "vizdoom_success": 0.0,
        "vizdoom_steps": 25.0,
        "vizdoom_invalid_actions": 1.0,
    }


def test_task_yaml_component_factories_are_wired():
    from lmms_eval.agentic import ActionParser, EnvManager, ObservationParser
    from lmms_eval.agentic.components import resolve

    env = resolve("env_manager", vizdoom_utils.vizdoom_env_manager, expected=EnvManager, doc={"instruction": "x"}, lmms_eval_specific_kwargs=None)
    obs = resolve("observation_parser", vizdoom_utils.vizdoom_observation_parser, expected=ObservationParser, doc={}, lmms_eval_specific_kwargs=None)
    act = resolve("action_parser", vizdoom_utils.vizdoom_action_parser, expected=ActionParser, doc={}, lmms_eval_specific_kwargs=None)

    assert type(env).__name__ == "VizDoomEnvManager"
    assert type(obs).__name__ == "VizDoomObservationParser"
    assert type(act).__name__ == "VizDoomActionParser"

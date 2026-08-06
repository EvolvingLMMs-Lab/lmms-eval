"""VizDoom task components exercised without the vizdoom binary."""

from __future__ import annotations

import json

from lmms_eval.agentic import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvManager,
    EnvState,
    ParserContext,
)
from lmms_eval.agentic.components import resolve
from lmms_eval.agentic.pipelines import apply_parser_pipeline, select_parser_pipelines
from lmms_eval.tasks.vizdoom_agentic import utils as vizdoom_utils
from lmms_eval.utils import load_yaml_config


def _ctx(state, agent_id="agent"):
    return ParserContext(state=state, agent_id=agent_id, step_idx=state.step_idx)


def _state(observation=None):
    return EnvState(env_id="vizdoom", step_idx=0, observation=observation or {})


def _parser_module():
    return vizdoom_utils._sibling_module("parsers")


def _action_state(buttons):
    return _state({"available_buttons": buttons})


def test_action_parser_reads_plain_button_combo():
    state = _action_state(["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    parsed = _parser_module().vizdoom_action_parser(AgentOutput(content=[ContentBlock.text("MOVE_LEFT + ATTACK")]), _ctx(state))

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": ["MOVE_LEFT", "ATTACK"]}


def test_action_parser_reads_json_button_values():
    state = _action_state(["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    parsed = _parser_module().vizdoom_action_parser(AgentOutput(content=[ContentBlock.text('{"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}')]), _ctx(state))

    assert parsed.action.data == {"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}


def test_action_parser_errors_on_gibberish():
    state = _action_state(["ATTACK"])
    parsed = _parser_module().vizdoom_action_parser(AgentOutput(content=[ContentBlock.text("no button here")]), _ctx(state))

    assert parsed.action is None
    assert parsed.error is not None


def test_observation_parser_human_view_emits_prompt_only_text():
    state = _state({"screen_buffer": None, "game_variables": {"HEALTH": 100}})

    request = _parser_module().vizdoom_observation_parser(state, _ctx(state))

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


def test_task_yaml_model_specific_parser_functions_are_wired():
    env = resolve("env_manager", vizdoom_utils.vizdoom_env_manager, expected=EnvManager, doc={"instruction": "x"}, lmms_eval_specific_kwargs=None)
    config = load_yaml_config("lmms_eval/tasks/vizdoom_agentic/vizdoom.yaml")
    selected = select_parser_pipelines(config["model_specific_parsers"], "Qwen/Qwen3.6-27B")

    assert type(env).__name__ == "VizDoomEnvManager"
    assert callable(selected["observation"])
    assert [parser.__name__ for parser in selected["action"]] == ["vizdoom_qwen_output_parser", "vizdoom_action_parser"]


def test_qwen_pipeline_strips_thinking_and_extracts_tool_call():
    state = _action_state(["ATTACK"])
    context = _ctx(state)
    text = "<think>aim</think><tool_call><function=press_buttons><parameter=buttons>ATTACK</parameter></function></tool_call>"
    pipeline = [_parser_module().vizdoom_qwen_output_parser, _parser_module().vizdoom_action_parser]

    parsed = apply_parser_pipeline(AgentOutput(content=[ContentBlock.text(text)]), pipeline, context)

    assert parsed.action.type == "ATTACK"
    assert parsed.metadata["skill_call"]["name"] == "press_buttons"

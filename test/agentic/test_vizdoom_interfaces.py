from __future__ import annotations

import pytest

from lmms_eval.agentic.model_server.lmms import _doc_to_messages, _split_agent_input
from lmms_eval.agentic.parsers.action.vizdoom_vllm_parser import VizDoomVllmActionParser
from lmms_eval.agentic.parsers.observation.vizdoom_vllm_parser import (
    VizDoomVllmObservationParser,
)
from lmms_eval.agentic.registry import build_action_parser, build_observation_parser
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock, EnvState
from lmms_eval.tasks.vizdoom_agentic.env import VizDoomEnv


def test_vizdoom_action_parser_reads_plain_button_combo():
    parser = VizDoomVllmActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    state = EnvState(env_id="vizdoom", step_idx=0, observation={})

    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("MOVE_LEFT + ATTACK")]), state, agent_id="agent")

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": ["MOVE_LEFT", "ATTACK"]}


def test_vizdoom_vllm_parser_registry_name_builds_both_parser_types():
    observation_parser = build_observation_parser({"name": "vizdoom_vllm_parser", "video": True})
    action_parser = build_action_parser({"name": "vizdoom_vllm_parser", "buttons": ["ATTACK"]})

    assert isinstance(observation_parser, VizDoomVllmObservationParser)
    assert isinstance(action_parser, VizDoomVllmActionParser)


def test_vizdoom_action_parser_reads_json_button_values():
    parser = VizDoomVllmActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    state = EnvState(env_id="vizdoom", step_idx=0, observation={})

    parsed = parser.parse(AgentOutput(content=[ContentBlock.text('{"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}')]), state)

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}


def test_vizdoom_action_parser_reads_skill_call_metadata():
    parser = VizDoomVllmActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    state = EnvState(env_id="vizdoom", step_idx=0, observation={})

    parsed = parser.parse(
        AgentOutput(
            content=[ContentBlock.text("calling a skill")],
            metadata={"tool_calls": [{"name": "press_buttons", "arguments": {"buttons": "MOVE_LEFT, ATTACK", "tics": "3"}}]},
        ),
        state,
    )

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": ["MOVE_LEFT", "ATTACK"], "tics": 3}


def test_vizdoom_action_parser_reads_qwen_xml_skill_call():
    parser = VizDoomVllmActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    state = EnvState(env_id="vizdoom", step_idx=0, observation={})

    parsed = parser.parse(
        AgentOutput(content=[ContentBlock.text("<tool_call>\n" "<function=press_buttons>\n" "<parameter=buttons>\nMOVE_RIGHT, ATTACK\n</parameter>\n" "<parameter=tics>\n2\n</parameter>\n" "</function>\n" "</tool_call>")]),
        state,
    )

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": ["MOVE_RIGHT", "ATTACK"], "tics": 2}


def test_vizdoom_action_parser_reads_function_style_skill_call():
    parser = VizDoomVllmActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    state = EnvState(env_id="vizdoom", step_idx=0, observation={})

    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("press_buttons(MOVE_LEFT, ATTACK, tics=4)")]), state)

    assert parsed.action is not None
    assert parsed.action.type == "vizdoom_action"
    assert parsed.action.data == {"buttons": ["MOVE_LEFT", "ATTACK"], "tics": 4}


def test_vizdoom_action_parser_maps_shooting_text_to_attack():
    parser = VizDoomVllmActionParser(buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    state = EnvState(env_id="vizdoom", step_idx=0, observation={})

    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("The monster is visible, so continue shooting.")]), state)

    assert parsed.action is not None
    assert parsed.action.type == "ATTACK"


def test_lmms_bridge_preserves_video_media_type():
    request = AgentInput(content=[ContentBlock.text("play"), ContentBlock(type="video", data=["frame0", "frame1"])])

    context, media = _split_agent_input(request)
    messages = _doc_to_messages(context, media)({})

    assert messages[0]["content"][0] == {"type": "video", "url": ["frame0", "frame1"]}
    assert messages[0]["content"][1] == {"type": "text", "text": "play"}


def test_vizdoom_observation_parser_can_emit_video_and_state_blocks():
    pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    state = EnvState(
        env_id="vizdoom",
        step_idx=1,
        observation={
            "instruction": "shoot",
            "step_idx": 1,
            "available_buttons": ["ATTACK"],
            "screen_buffer": frame,
            "screen_history": [frame, frame],
            "game_variables": {"AMMO2": 50.0},
        },
    )
    parser = VizDoomVllmObservationParser(video=True, image_buffers=["screen"])

    request = parser.parse(state)

    assert [block.type for block in request.content[:3]] == ["text", "video", "image"]
    assert any(block.type == "vizdoom_state" for block in request.content)
    assert "press_buttons" in request.first_text()
    assert "<think>...</think>" in request.first_text()
    assert "<parameter=tics>12</parameter>" in request.first_text()
    assert "Do not answer with JSON" in request.first_text()


def test_vizdoom_observation_parser_can_disable_thinking_prompt():
    state = EnvState(
        env_id="vizdoom",
        step_idx=1,
        observation={
            "instruction": "shoot",
            "step_idx": 1,
            "available_buttons": ["ATTACK"],
        },
    )
    parser = VizDoomVllmObservationParser(require_thinking=False)

    request = parser.parse(state)

    assert "<think>...</think>" not in request.first_text()
    assert "press_buttons" in request.first_text()


def test_vizdoom_native_env_exposes_buffers_and_metadata():
    pytest.importorskip("vizdoom")
    env = VizDoomEnv(
        config_path="basic.cfg",
        screen_resolution="RES_160X120",
        screen_format="RGB24",
        available_buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        available_game_variables=["AMMO2", "HEALTH", "KILLCOUNT"],
        depth_buffer=True,
        labels_buffer=True,
        automap_buffer=True,
        objects_info=True,
        sectors_info=True,
        notifications_buffer=True,
        notifications_buffer_size=4,
        window_visible=False,
        sound_enabled=False,
        frame_history=3,
        tics_per_action=1,
        capture_action_frames=False,
    )
    try:
        state = env.reset({"id": "basic", "instruction": "shoot"}, seed=1)
        observation = state.observation

        assert observation["screen_buffer"].shape == (120, 160, 3)
        assert observation["depth_buffer"].shape == (120, 160)
        assert observation["labels_buffer"].shape == (120, 160)
        assert observation["automap_buffer"].shape == (120, 160, 3)
        assert "AMMO2" in observation["game_variables"]
        assert "HEALTH" in observation["game_variables"]
        assert isinstance(observation["objects"], list)
        assert isinstance(observation["sectors"], list)
        assert len(observation["screen_history"]) == 1

        parsed = VizDoomVllmActionParser(buttons=["ATTACK"]).parse(AgentOutput(content=[ContentBlock.text("ATTACK")]), state)
        assert parsed.action is not None
        result = env.step(parsed.action)
        assert result.info["action_vector"] == [0.0, 0.0, 1.0]
        assert len(result.state.observation["screen_history"]) == 2
    finally:
        env.close()


def test_vizdoom_native_env_can_capture_frames_within_long_action():
    pytest.importorskip("vizdoom")
    env = VizDoomEnv(
        config_path="basic.cfg",
        screen_resolution="RES_160X120",
        screen_format="RGB24",
        available_buttons=["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        available_game_variables=["AMMO2", "HEALTH", "KILLCOUNT"],
        window_visible=False,
        sound_enabled=False,
        frame_history=12,
        tics_per_action=12,
        capture_action_frames=True,
    )
    try:
        state = env.reset({"id": "basic", "instruction": "shoot"}, seed=1)
        parsed = VizDoomVllmActionParser(buttons=["ATTACK"]).parse(AgentOutput(content=[ContentBlock.text("ATTACK")]), state)

        result = env.step(parsed.action)

        assert result.info["tics"] == 12
        assert result.info["elapsed_tics"] >= 1
        assert result.info["captured_action_frames"] >= 1
        assert result.state.observation["step_idx"] == 1
        assert 1 <= len(result.state.observation["screen_history"]) <= 12
    finally:
        env.close()

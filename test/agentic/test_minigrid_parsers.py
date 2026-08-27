"""Task-local MiniGrid parser functions."""

from __future__ import annotations

from lmms_eval.agentic import (
    ActionDef,
    ActionSpec,
    AgentOutput,
    ContentBlock,
    EnvState,
    ParserContext,
)
from lmms_eval.tasks.minigrid_agentic.parsers import (
    minigrid_action_parser,
    minigrid_observation_parser,
    minigrid_qwen_output_parser,
)


def _spec():
    return ActionSpec(
        kind="discrete",
        actions=[ActionDef(name="LEFT", description="turn left", aliases=["TURN_LEFT"]), ActionDef(name="FORWARD")],
    )


def _state(observation, step_idx=0):
    return EnvState(env_id="minigrid", step_idx=step_idx, observation=observation)


def _context(state=None, *, doc=None, spec=None, max_steps=8):
    return ParserContext(state=state, agent_id="agent", step_idx=0, metadata={"max_steps": max_steps, "doc": doc, "action_spec": spec})


def _output(text, **metadata):
    return AgentOutput(content=[ContentBlock.text(text)], metadata=metadata)


def test_observation_parser_renders_task_state_and_action_spec():
    state = _state({"text": "get to the goal", "variables": {"facing": "east"}})

    request = minigrid_observation_parser(state, _context(state, doc={"instruction": "Play well."}, spec=_spec()))

    text = request.first_text()
    assert "Play well." in text
    assert "get to the goal" in text
    assert 'Variables: {"facing": "east"}' in text
    assert "Step 0 of 8." in text
    assert "- LEFT: turn left" in text
    assert "Respond with only the action name." in text


def test_observation_parser_preserves_image_and_video_values():
    state = _state({"text": "obs", "video": ["f0", "f1"], "images": ["i0", "i1"]})

    request = minigrid_observation_parser(state, _context(state))

    blocks = [(block.type, block.data) for block in request.content[1:]]
    assert blocks == [("video", ["f0", "f1"]), ("image", "i0"), ("image", "i1")]


def test_action_parser_reads_plain_names_aliases_json_and_tool_calls():
    context = _context(spec=_spec())

    assert minigrid_action_parser(_output("FORWARD"), context).action.type == "FORWARD"
    assert minigrid_action_parser(_output("turn_left"), context).action.type == "LEFT"
    assert minigrid_action_parser(_output('{"action": "LEFT"}'), context).action.type == "LEFT"
    assert minigrid_action_parser(_output("", tool_calls=[{"name": "FORWARD", "arguments": {}}]), context).action.type == "FORWARD"


def test_action_parser_reports_invalid_output():
    parsed = minigrid_action_parser(_output("DANCE"), _context(spec=_spec()))

    assert parsed.action is None
    assert parsed.error == "no valid MiniGrid action found"


def test_qwen_output_parser_removes_completed_thinking_block():
    output = minigrid_qwen_output_parser(_output("<think>LEFT is risky</think>FORWARD"), _context())

    assert output.first_text() == "FORWARD"
    assert output.metadata["raw_text"] == "<think>LEFT is risky</think>FORWARD"

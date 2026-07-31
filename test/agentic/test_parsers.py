from __future__ import annotations

from lmms_eval.agentic import AgentOutput, ContentBlock, EnvState
from lmms_eval.agentic.parsers import (
    ActionNameParser,
    ParserContext,
    QwenModelOutputParser,
)


def _ctx(agent_id="agent"):
    return ParserContext(state=EnvState(env_id="test", step_idx=0, observation={}), agent_id=agent_id, step_idx=0)


def test_qwen_parser_strips_thinking_block():
    parser = QwenModelOutputParser()
    output = parser.parse(AgentOutput(content=[ContentBlock.text("<think>plan...</think>ATTACK")]), _ctx())

    assert output.first_text() == "ATTACK"
    assert output.metadata["raw_text"] == "<think>plan...</think>ATTACK"


def test_qwen_parser_keeps_reasoning_when_nothing_follows_think_close():
    parser = QwenModelOutputParser()
    output = parser.parse(AgentOutput(content=[ContentBlock.text("<think>I should ATTACK</think>")]), _ctx())

    assert "ATTACK" in output.first_text()


def test_qwen_parser_extracts_tool_calls():
    text = "<tool_call><function=press_buttons><parameter=buttons>ATTACK</parameter></function></tool_call>"
    parser = QwenModelOutputParser()
    output = parser.parse(AgentOutput(content=[ContentBlock.text(text)]), _ctx())

    assert output.metadata["tool_calls"] == [{"name": "press_buttons", "arguments": {"buttons": "ATTACK"}}]


def test_action_name_parser_reads_plain_text():
    parser = ActionNameParser(actions=["MOVE_LEFT", "ATTACK"])
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("I will attack now")]), _ctx())

    assert parsed.action.type == "ATTACK"
    assert parsed.is_submit is False


def test_action_name_parser_reads_json_and_aliases():
    parser = ActionNameParser(actions=["MOVE_LEFT"], aliases={"LEFT": "MOVE_LEFT"})
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text('{"action": "left"}')]), _ctx())

    assert parsed.action.type == "MOVE_LEFT"


def test_action_name_parser_flags_submit_actions():
    parser = ActionNameParser(actions=["ATTACK"], submit_actions=["SUBMIT"])
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("SUBMIT")]), _ctx())

    assert parsed.is_submit is True


def test_action_name_parser_errors_on_unknown_action():
    parser = ActionNameParser(actions=["ATTACK"])
    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("dance")]), _ctx())

    assert parsed.action is None
    assert parsed.error == "no valid action found"

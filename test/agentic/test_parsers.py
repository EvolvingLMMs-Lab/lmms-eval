from __future__ import annotations

from lmms_eval.agentic.parsers import ActionNameParser, QwenModelOutputParser
from lmms_eval.agentic.types import AgentOutput, ContentBlock, EnvState


def _state():
    return EnvState(env_id="test", step_idx=0, observation={})


def test_qwen_model_output_parser_strips_closed_thinking_block():
    parser = QwenModelOutputParser()

    output = parser.parse(AgentOutput(content=[ContentBlock.text("<think>\nreason\n</think>\n\nMOVE_FORWARD")]), _state())

    assert output.first_text() == "MOVE_FORWARD"
    assert output.metadata["raw_text"] == "<think>\nreason\n</think>\n\nMOVE_FORWARD"
    assert output.metadata["normalized_text"] == "MOVE_FORWARD"


def test_qwen_model_output_parser_keeps_thinking_when_no_final_answer():
    parser = QwenModelOutputParser()

    output = parser.parse(AgentOutput(content=[ContentBlock.text("The monster is visible, so attack now.\\n</think>")]), _state())

    assert "attack now" in output.first_text()
    assert output.metadata["normalized_text"] == output.first_text()


def test_action_name_parser_reads_plain_action():
    parser = ActionNameParser(actions=["MOVE_FORWARD", "NOOP"])

    parsed = parser.parse(AgentOutput(content=[ContentBlock.text("MOVE_FORWARD")]), _state(), agent_id="agent")

    assert parsed.action is not None
    assert parsed.action.type == "MOVE_FORWARD"
    assert parsed.action.agent_id == "agent"


def test_action_name_parser_reads_tool_call_metadata():
    parser = ActionNameParser(actions=["MOVE_FORWARD", "NOOP"])

    parsed = parser.parse(
        AgentOutput(content=[ContentBlock.text("call tool")], metadata={"tool_calls": [{"name": "act", "arguments": {"action": "NOOP"}}]}),
        _state(),
    )

    assert parsed.action is not None
    assert parsed.action.type == "NOOP"


def test_action_name_parser_reads_qwen_xml_parameter():
    parser = ActionNameParser(actions=["MOVE_FORWARD", "NOOP"])

    parsed = parser.parse(
        AgentOutput(content=[ContentBlock.text("<tool_call>\n<function=act>\n<parameter=action>\nMOVE_FORWARD\n</parameter>\n</function>\n</tool_call>")]),
        _state(),
    )

    assert parsed.action is not None
    assert parsed.action.type == "MOVE_FORWARD"

"""ActionSpec rendering and the spec-driven action parsers."""

from __future__ import annotations

import pytest

from lmms_eval.agentic import (
    ActionDef,
    ActionNameParser,
    ActionSpec,
    AgentOutput,
    ContentBlock,
    FreeTextActionParser,
    SchemaActionParser,
    build_action_parser,
)
from lmms_eval.agentic.parsers import ParserContext


def _output(text: str, **metadata):
    return AgentOutput(content=[ContentBlock.text(text)], metadata=metadata)


def _ctx():
    return ParserContext(agent_id="agent", step_idx=0)


def _goto_spec(**overrides):
    schema = {"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}
    kwargs = {
        "kind": "parameterized",
        "actions": [ActionDef(name="GOTO", description="move to a cell", schema=schema), ActionDef(name="STOP", description="end the episode")],
        "submit_actions": ["STOP"],
    }
    kwargs.update(overrides)
    return ActionSpec(**kwargs)


def test_action_def_render_includes_parameter_signature():
    action = ActionDef(name="GOTO", description="move to a cell", schema={"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}})

    assert action.render() == "GOTO(x: integer, y: integer): move to a cell"


def test_action_spec_render_prompt_lists_actions_and_hint():
    spec = ActionSpec(kind="discrete", actions=[ActionDef(name="LEFT", description="turn left"), ActionDef(name="RIGHT")], prompt_hint="One action per turn.")

    rendered = spec.render_prompt()

    assert "- LEFT: turn left" in rendered
    assert "- RIGHT" in rendered
    assert rendered.endswith("One action per turn.")


def test_action_name_parser_from_spec_honors_aliases_and_submit():
    spec = ActionSpec(
        kind="discrete",
        actions=[ActionDef(name="LEFT", aliases=["TURN_LEFT"]), ActionDef(name="STOP")],
        submit_actions=["STOP"],
    )
    parser = ActionNameParser.from_spec(spec)

    parsed = parser.parse(_output("turn_left"), _ctx())
    assert parsed.action.type == "LEFT"

    submitted = parser.parse(_output("STOP"), _ctx())
    assert submitted.is_submit is True


def test_action_name_parser_from_spec_has_no_implicit_submit():
    parser = ActionNameParser.from_spec(ActionSpec(kind="discrete", actions=[ActionDef(name="LEFT")]))

    parsed = parser.parse(_output("SUBMIT"), _ctx())

    assert parsed.action is None
    assert parsed.error is not None


def test_schema_parser_reads_tool_calls():
    parser = SchemaActionParser(_goto_spec())

    parsed = parser.parse(_output("", tool_calls=[{"name": "goto", "arguments": {"x": 1, "y": 2}}]), _ctx())

    assert parsed.action.type == "GOTO"
    assert parsed.action.data == {"x": 1, "y": 2}
    assert parsed.is_submit is False


def test_schema_parser_reads_inline_json_with_flat_arguments():
    parser = SchemaActionParser(_goto_spec())

    parsed = parser.parse(_output('I choose {"action": "GOTO", "x": 3, "y": 4}'), _ctx())

    assert parsed.action.type == "GOTO"
    assert parsed.action.data == {"x": 3, "y": 4}


def test_schema_parser_rejects_missing_required_argument():
    parser = SchemaActionParser(_goto_spec())

    parsed = parser.parse(_output('{"action": "GOTO", "x": 3}'), _ctx())

    assert parsed.action is None
    assert "y" in parsed.error


def test_schema_parser_rejects_wrong_argument_type():
    parser = SchemaActionParser(_goto_spec())

    parsed = parser.parse(_output('{"action": "GOTO", "x": "left", "y": 4}'), _ctx())

    assert parsed.action is None
    assert "x" in parsed.error


def test_schema_parser_matches_bare_names_for_zero_arg_actions():
    parser = SchemaActionParser(_goto_spec())

    parsed = parser.parse(_output("I will STOP now."), _ctx())

    assert parsed.action.type == "STOP"
    assert parsed.is_submit is True


def test_free_text_parser_wraps_whole_reply():
    parser = FreeTextActionParser(submit_actions=["QUIT"])

    parsed = parser.parse(_output("  go north  "), _ctx())
    assert parsed.action.type == "text_command"
    assert parsed.action.data == "go north"
    assert parsed.is_submit is False

    assert parser.parse(_output("quit"), _ctx()).is_submit is True
    assert parser.parse(_output("   "), _ctx()).error is not None


@pytest.mark.parametrize(
    "kind, expected_type",
    [("discrete", ActionNameParser), ("parameterized", SchemaActionParser), ("free_text", FreeTextActionParser)],
)
def test_build_action_parser_dispatches_on_kind(kind, expected_type):
    spec = ActionSpec(kind=kind, actions=[ActionDef(name="STOP")])

    assert isinstance(build_action_parser(spec), expected_type)


def test_build_action_parser_requires_a_spec():
    with pytest.raises(ValueError, match="action_spec"):
        build_action_parser(None)

    with pytest.raises(ValueError, match="Unknown ActionSpec.kind"):
        build_action_parser(ActionSpec(kind="telepathy"))

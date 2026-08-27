"""Environment-declared ActionSpec data and rendering behavior."""

from __future__ import annotations

from lmms_eval.agentic import ActionDef, ActionSpec


def test_action_def_render_includes_parameter_signature():
    action = ActionDef(name="GOTO", description="move to a cell", schema={"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}})

    assert action.render() == "GOTO(x: integer, y: integer): move to a cell"


def test_action_spec_render_prompt_lists_actions_and_hint():
    spec = ActionSpec(kind="discrete", actions=[ActionDef(name="LEFT", description="turn left"), ActionDef(name="RIGHT")], prompt_hint="One action per turn.")

    rendered = spec.render_prompt()

    assert "- LEFT: turn left" in rendered
    assert "- RIGHT" in rendered
    assert rendered.endswith("One action per turn.")


def test_action_spec_exposes_names_aliases_and_lookup():
    spec = ActionSpec(actions=[ActionDef(name="LEFT", aliases=["TURN_LEFT"]), ActionDef(name="FORWARD")])

    assert spec.action_names() == ["LEFT", "FORWARD"]
    assert spec.alias_map() == {"TURN_LEFT": "LEFT"}
    assert spec.get("left").name == "LEFT"
    assert spec.get("missing") is None


def test_free_text_spec_without_actions_renders_default_hint():
    spec = ActionSpec(kind="free_text")

    assert spec.render_prompt() == "Respond with a single short text command."

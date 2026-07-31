"""TemplateObservationParser: reserved observation keys -> AgentInput."""

from __future__ import annotations

from lmms_eval.agentic import ActionDef, ActionSpec, EnvState, TemplateObservationParser
from lmms_eval.agentic.parsers import ParserContext


def _state(observation, step_idx=0):
    return EnvState(env_id="env", step_idx=step_idx, observation=observation)


def _ctx(doc=None, spec=None, max_steps=8):
    metadata = {"max_steps": max_steps, "doc": doc, "action_spec": spec}
    return ParserContext(agent_id="agent", step_idx=0, metadata=metadata)


def _spec():
    return ActionSpec(kind="discrete", actions=[ActionDef(name="LEFT", description="turn left"), ActionDef(name="FORWARD")])


def test_default_template_renders_all_reserved_sections():
    parser = TemplateObservationParser()
    state = _state({"text": "get to the goal", "variables": {"facing": "east"}})

    request = parser.parse(state, _ctx(doc={"instruction": "Play well."}, spec=_spec()))

    text = request.first_text()
    assert "Play well." in text
    assert "get to the goal" in text
    assert 'Variables: {"facing": "east"}' in text
    assert "Step 0 of 8." in text
    assert "- LEFT: turn left" in text
    assert "Respond with only the action name." in text


def test_default_template_drops_empty_sections():
    parser = TemplateObservationParser()

    text = parser.parse(_state({"text": "hello"}), _ctx()).first_text()

    assert "hello" in text
    assert "Variables" not in text
    assert "Available actions" not in text
    assert "Respond with" not in text
    assert "Step 0 of 8." in text


def test_custom_template_with_unknown_placeholder_renders_empty():
    parser = TemplateObservationParser(template="{instruction} | {text} | {foo} | {step_idx}/{max_steps}")
    state = _state({"text": "obs"}, step_idx=3)

    text = parser.parse(state, _ctx(doc={"instruction": "Go"})).first_text()

    assert text == "Go | obs |  | 3/8"


def test_observation_actions_text_overrides_spec_rendering():
    parser = TemplateObservationParser()
    state = _state({"text": "obs", "actions": "custom action help"})

    text = parser.parse(state, _ctx(spec=_spec())).first_text()

    assert "custom action help" in text
    assert "- LEFT" not in text


def test_media_blocks_from_reserved_keys():
    parser = TemplateObservationParser(max_images=2)
    state = _state({"text": "obs", "video": ["f0", "f1"], "images": ["i0", "i1", "i2"]})

    request = parser.parse(state, _ctx())

    blocks = {(block.type, tuple(block.data) if isinstance(block.data, list) else block.data) for block in request.content[1:]}
    assert ("video", ("f0", "f1")) in blocks
    assert ("image", "i1") in blocks and ("image", "i2") in blocks
    assert ("image", "i0") not in blocks


def test_media_blocks_can_be_disabled():
    parser = TemplateObservationParser(include_images=False, include_video=False)
    state = _state({"text": "obs", "video": ["f0"], "images": ["i0"]})

    request = parser.parse(state, _ctx())

    assert [block.type for block in request.content] == ["text"]


def test_non_dict_observation_becomes_text():
    parser = TemplateObservationParser()

    text = parser.parse(_state("raw state"), _ctx()).first_text()

    assert "raw state" in text

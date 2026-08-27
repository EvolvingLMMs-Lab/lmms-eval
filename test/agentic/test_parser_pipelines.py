from __future__ import annotations

import pytest

from lmms_eval.agentic import (
    ParserContext,
    apply_parser_pipeline,
    select_parser_pipelines,
)


def test_parser_pipeline_is_any_to_any_and_runs_in_order():
    seen = []
    context = ParserContext(model_name="Qwen/Qwen3.6-27B")

    def mapping_to_number(value, ctx):
        seen.append(ctx.model_name)
        return value["number"]

    def number_to_text(value, ctx):
        seen.append(ctx.model_name)
        return f"action-{value}"

    assert apply_parser_pipeline({"number": 7}, [mapping_to_number, number_to_text], context) == "action-7"
    assert seen == ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B"]


def test_model_glob_overrides_one_default_pipeline():
    default_observation = lambda value, ctx: value
    default_action = lambda value, ctx: value
    qwen_action = lambda value, ctx: value
    config = {
        "default": {"observation": default_observation, "action": default_action},
        "*Qwen*": {"action": qwen_action},
    }

    selected = select_parser_pipelines(config, "/models/Qwen3.6-27B")

    assert selected == {"observation": default_observation, "action": qwen_action}


def test_exact_model_key_wins_over_earlier_glob():
    identity = lambda value, ctx: value
    glob_action = lambda value, ctx: "glob"
    exact_action = lambda value, ctx: "exact"
    config = {
        "default": {"observation": identity, "action": identity},
        "*qwen*": {"action": glob_action},
        "Qwen/Qwen3.6-27B": {"action": exact_action},
    }

    selected = select_parser_pipelines(config, "qwen/qwen3.6-27b")

    assert selected["action"] is exact_action


def test_default_pipelines_work_without_model_identity():
    identity = lambda value, ctx: value
    selected = select_parser_pipelines({"default": {"observation": identity, "action": identity}}, None)

    assert selected == {"observation": identity, "action": identity}


def test_selection_requires_both_boundary_pipelines():
    with pytest.raises(ValueError, match="action"):
        select_parser_pipelines({"default": {"observation": lambda value, ctx: value}}, "model")


def test_pipeline_rejects_non_callable_entries():
    with pytest.raises(TypeError, match="callable"):
        apply_parser_pipeline("value", ["not-a-function"], ParserContext())

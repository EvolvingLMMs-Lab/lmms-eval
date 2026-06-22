from __future__ import annotations

from types import SimpleNamespace

from lmms_eval.agentic.factory import AgenticFactory
from lmms_eval.agentic.loop.runner import (
    _runtime_agentic_factory,
    _runtime_component_spec,
    _runtime_max_parallel_rollouts,
    _rollout_plan_from_request,
    _split_model_server_loop_args,
)
from lmms_eval.api.instance import Instance


def test_runtime_component_spec_uses_default_without_cli_args():
    assert _runtime_component_spec(None, "agentic_model_output_parser", "agentic_model_output_parser_args", "identity") == "identity"


def test_runtime_component_spec_uses_cli_name():
    cli_args = SimpleNamespace(agentic_model_output_parser="qwen", agentic_model_output_parser_args="")

    assert _runtime_component_spec(cli_args, "agentic_model_output_parser", "agentic_model_output_parser_args", "identity") == "qwen"


def test_runtime_component_spec_combines_cli_name_and_args():
    cli_args = SimpleNamespace(
        agentic_model_server="openai",
        agentic_model_server_args="model=Qwen/Qwen3.5-9B,max_parallel_rollouts=4",
    )

    assert _runtime_component_spec(cli_args, "agentic_model_server", "agentic_model_server_args", "openai") == {
        "name": "openai",
        "model": "Qwen/Qwen3.5-9B",
        "max_parallel_rollouts": 4,
    }


def test_legacy_model_server_parallel_rollouts_moves_to_loop_manager_config():
    spec = {
        "name": "openai",
        "model": "Qwen/Qwen3.5-9B",
        "max_parallel_rollouts": 4,
    }

    model_server_spec, max_parallel_rollouts = _split_model_server_loop_args(spec)

    assert model_server_spec == {
        "name": "openai",
        "model": "Qwen/Qwen3.5-9B",
        "max_concurrent_requests": 4,
    }
    assert max_parallel_rollouts == 4


def test_agentic_max_parallel_rollouts_overrides_legacy_model_server_arg():
    cli_args = SimpleNamespace(agentic_max_parallel_rollouts=2)

    assert _runtime_max_parallel_rollouts(cli_args, default=4) == 2


def test_runtime_agentic_factory_accepts_custom_factory():
    factory = AgenticFactory().with_components(action_parsers={"custom": object})

    assert _runtime_agentic_factory(SimpleNamespace(agentic_factory=factory)) is factory
    assert factory.action_parsers["custom"] is object


def test_rollout_plan_uses_cli_observation_and_action_parsers_when_yaml_omits_them():
    req = Instance(
        request_type="generate_until_game",
        arguments=(
            "prompt",
            {"max_game_steps": 1},
            lambda doc: [],
            lambda: None,
            None,
            None,
            {},
            0,
            "task",
            "test",
        ),
        idx=0,
        metadata={"task": "task", "doc_id": 0, "repeats": 0},
    )
    lm = SimpleNamespace(task_dict={"task": {"test": [{"id": "doc"}]}})
    cli_args = SimpleNamespace(
        agentic_model_server=None,
        agentic_model_server_args="",
        agentic_loop_worker=None,
        agentic_loop_worker_args="",
        agentic_model_output_parser=None,
        agentic_model_output_parser_args="",
        agentic_observation_parser="vizdoom",
        agentic_observation_parser_args='video=true,image_buffers=["screen"]',
        agentic_action_parser="vizdoom",
        agentic_action_parser_args='submit_actions=["SUBMIT"],noop_actions=["NOOP"]',
        agentic_max_parallel_rollouts=None,
    )

    plan = _rollout_plan_from_request(0, lm, req, cli_args)

    assert plan.observation_parser == {
        "name": "vizdoom",
        "video": True,
        "image_buffers": ["screen"],
    }
    assert plan.action_parser == {
        "name": "vizdoom",
        "submit_actions": ["SUBMIT"],
        "noop_actions": ["NOOP"],
    }

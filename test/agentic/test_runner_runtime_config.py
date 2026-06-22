from __future__ import annotations

from types import SimpleNamespace

from lmms_eval.agentic.loop.runner import (
    _runtime_component_spec,
    _runtime_max_parallel_rollouts,
    _split_model_server_loop_args,
)


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

from __future__ import annotations

from typing import Any

from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.loop.worker import LoopWorker, SimpleLoopWorker
from lmms_eval.agentic.model_server import (
    ModelServer,
    OpenAIModelServer,
)
from lmms_eval.agentic.parsers import (
    ActionParser,
    ModelOutputParser,
    ObservationParser,
)
from lmms_eval.agentic.registry_core import (
    ACTION_PARSER_REGISTRY,
    LOOP_WORKER_REGISTRY,
    MODEL_OUTPUT_PARSER_REGISTRY,
    MODEL_SERVER_REGISTRY,
    OBSERVATION_PARSER_REGISTRY,
    call_factory,
    register_action_parser,
    register_loop_worker,
    register_model_output_parser,
    register_model_server,
    register_observation_parser,
)

register_model_server("openai", OpenAIModelServer, replace=True)
register_loop_worker("simple", SimpleLoopWorker, replace=True)


def build_model_server(spec: Any, **kwargs) -> ModelServer:
    return MODEL_SERVER_REGISTRY.build(spec or "openai", ModelServer, **kwargs)


def build_loop_worker(spec: Any, **kwargs) -> LoopWorker:
    return LOOP_WORKER_REGISTRY.build(spec or "simple", LoopWorker, **kwargs)


def build_env_manager(spec: Any, **kwargs) -> EnvManager:
    if isinstance(spec, EnvManager):
        return spec
    if spec is None:
        raise TypeError("EnvManager spec is required; in YAML use `game_env: !function utils.<factory>`")

    factory = spec
    if isinstance(spec, dict):
        spec_kwargs = dict(spec)
        factory = spec_kwargs.pop("factory", None)
        if factory is None:
            raise TypeError(
                "EnvManager dict specs require a callable `factory`; registry names are not supported. "
                "In YAML use `game_env: !function utils.<factory>`."
            )
        kwargs = {**kwargs, **spec_kwargs}

    if not callable(factory):
        raise TypeError(f"Expected EnvManager instance or callable factory; got {type(spec).__name__}")

    component = call_factory(factory, kwargs)
    if not isinstance(component, EnvManager):
        raise TypeError(f"EnvManager factory returned {type(component).__name__}, expected EnvManager")
    return component


def build_observation_parser(spec: Any, **kwargs) -> ObservationParser:
    return OBSERVATION_PARSER_REGISTRY.build(spec, ObservationParser, **kwargs)


def build_model_output_parser(spec: Any, **kwargs) -> ModelOutputParser:
    return MODEL_OUTPUT_PARSER_REGISTRY.build(spec or "identity", ModelOutputParser, **kwargs)


def build_action_parser(spec: Any, **kwargs) -> ActionParser:
    return ACTION_PARSER_REGISTRY.build(spec, ActionParser, **kwargs)

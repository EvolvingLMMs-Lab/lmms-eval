from __future__ import annotations

import inspect
from typing import Any

from lmms_eval.agentic.env import GameEnv
from lmms_eval.agentic.loop.worker import LoopWorker, SimpleLoopWorker
from lmms_eval.agentic.model_server import (
    ModelServer,
    OpenAIModelServer,
)
from lmms_eval.agentic.parsers import (
    ActionNameParser,
    ActionParser,
    IdentityModelOutputParser,
    ModelOutputParser,
    ObservationParser,
    QwenModelOutputParser,
    VizDoomActionParser,
    VizDoomObservationParser,
    VizDoomVllmActionParser,
    VizDoomVllmObservationParser,
)


class AgenticRegistry:
    """Small named registry for agentic components."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, value: Any = None, *, replace: bool = False):
        def add(item):
            if not replace and name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = item
            return item

        return add(value) if value is not None else add

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}") from exc

    def build(self, spec: Any, expected_type: type, **kwargs):
        if isinstance(spec, expected_type):
            return spec

        spec_kwargs = {}
        if isinstance(spec, dict):
            spec = dict(spec)
            name = spec.pop("name", None) or spec.pop("type", None) or spec.pop("id", None)
            if not name:
                raise TypeError(f"{self.kind} dict spec requires a 'name', 'type', or 'id' field")
            factory = self.get(name)
            spec_kwargs = spec
        else:
            factory = self.get(spec) if isinstance(spec, str) else spec

        if not callable(factory):
            raise TypeError(f"Expected {self.kind} name, instance, or factory; got {type(spec).__name__}")

        component = _call_factory(factory, {**kwargs, **spec_kwargs})
        if not isinstance(component, expected_type):
            raise TypeError(f"{self.kind} factory returned {type(component).__name__}, expected {expected_type.__name__}")
        return component

    def names(self) -> list[str]:
        return sorted(self._items)


def _call_factory(factory, kwargs: dict[str, Any]):
    signature = inspect.signature(factory)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return factory(**kwargs)

    accepted_kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return factory(**accepted_kwargs)


MODEL_SERVER_REGISTRY = AgenticRegistry("model_server")
LOOP_WORKER_REGISTRY = AgenticRegistry("loop_worker")
GAME_ENV_REGISTRY = AgenticRegistry("game_env")
OBSERVATION_PARSER_REGISTRY = AgenticRegistry("observation_parser")
ACTION_PARSER_REGISTRY = AgenticRegistry("action_parser")
MODEL_OUTPUT_PARSER_REGISTRY = AgenticRegistry("model_output_parser")

register_model_server = MODEL_SERVER_REGISTRY.register
register_loop_worker = LOOP_WORKER_REGISTRY.register
register_game_env = GAME_ENV_REGISTRY.register
register_observation_parser = OBSERVATION_PARSER_REGISTRY.register
register_action_parser = ACTION_PARSER_REGISTRY.register
register_model_output_parser = MODEL_OUTPUT_PARSER_REGISTRY.register

register_model_server("openai", OpenAIModelServer, replace=True)
register_loop_worker("simple", SimpleLoopWorker, replace=True)
register_observation_parser("vizdoom_vllm_parser", VizDoomVllmObservationParser, replace=True)
register_observation_parser("vizdoom", VizDoomObservationParser, replace=True)
register_model_output_parser("identity", IdentityModelOutputParser, replace=True)
register_model_output_parser("qwen", QwenModelOutputParser, replace=True)
register_action_parser("action_name", ActionNameParser, replace=True)
register_action_parser("vizdoom_vllm_parser", VizDoomVllmActionParser, replace=True)
register_action_parser("vizdoom_action", VizDoomActionParser, replace=True)


def build_model_server(spec: Any, **kwargs) -> ModelServer:
    return MODEL_SERVER_REGISTRY.build(spec or "openai", ModelServer, **kwargs)


def build_loop_worker(spec: Any, **kwargs) -> LoopWorker:
    return LOOP_WORKER_REGISTRY.build(spec or "simple", LoopWorker, **kwargs)


def build_game_env(spec: Any, **kwargs) -> GameEnv:
    return GAME_ENV_REGISTRY.build(spec, GameEnv, **kwargs)


def build_observation_parser(spec: Any, **kwargs) -> ObservationParser:
    return OBSERVATION_PARSER_REGISTRY.build(spec, ObservationParser, **kwargs)


def build_model_output_parser(spec: Any, **kwargs) -> ModelOutputParser:
    return MODEL_OUTPUT_PARSER_REGISTRY.build(spec or "identity", ModelOutputParser, **kwargs)


def build_action_parser(spec: Any, **kwargs) -> ActionParser:
    return ACTION_PARSER_REGISTRY.build(spec, ActionParser, **kwargs)

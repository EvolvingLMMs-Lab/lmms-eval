from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.model_server import (
    FixedActionModelServer,
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
)


@dataclass(slots=True)
class AgenticFactory:
    """Explicit factory for agentic components.

    Built-in short names are local defaults, not global registration state.
    Callers can pass callables/import paths directly, or create a modified
    factory with ``with_components``.
    """

    model_servers: dict[str, Any] = field(default_factory=lambda: {"openai": OpenAIModelServer, "debug": FixedActionModelServer})
    loop_workers: dict[str, Any] = field(default_factory=lambda: {"simple": "lmms_eval.agentic.loop.worker.simple:SimpleLoopWorker"})
    model_output_parsers: dict[str, Any] = field(default_factory=lambda: {"identity": IdentityModelOutputParser, "qwen": QwenModelOutputParser})
    observation_parsers: dict[str, Any] = field(default_factory=lambda: {"vizdoom": VizDoomObservationParser})
    action_parsers: dict[str, Any] = field(default_factory=lambda: {"action_name": ActionNameParser, "vizdoom": VizDoomActionParser})

    def with_components(
        self,
        *,
        model_servers: dict[str, Any] | None = None,
        loop_workers: dict[str, Any] | None = None,
        model_output_parsers: dict[str, Any] | None = None,
        observation_parsers: dict[str, Any] | None = None,
        action_parsers: dict[str, Any] | None = None,
    ) -> "AgenticFactory":
        return AgenticFactory(
            model_servers={**self.model_servers, **(model_servers or {})},
            loop_workers={**self.loop_workers, **(loop_workers or {})},
            model_output_parsers={**self.model_output_parsers, **(model_output_parsers or {})},
            observation_parsers={**self.observation_parsers, **(observation_parsers or {})},
            action_parsers={**self.action_parsers, **(action_parsers or {})},
        )

    def build_model_server(self, spec: Any, **kwargs) -> ModelServer:
        return self._build(spec or "openai", ModelServer, self.model_servers, "model_server", **kwargs)

    def build_loop_worker(self, spec: Any, **kwargs):
        from lmms_eval.agentic.loop.worker.base import LoopWorker

        return self._build(spec or "simple", LoopWorker, self.loop_workers, "loop_worker", **kwargs)

    def build_env_manager(self, spec: Any, **kwargs) -> EnvManager:
        if spec is None:
            raise TypeError("EnvManager spec is required; in YAML use `game_env: !function utils.<factory>`")
        return self._build(spec, EnvManager, {}, "env_manager", require_factory_for_dict=True, **kwargs)

    def build_observation_parser(self, spec: Any, **kwargs) -> ObservationParser:
        return self._build(spec, ObservationParser, self.observation_parsers, "observation_parser", **kwargs)

    def build_model_output_parser(self, spec: Any, **kwargs) -> ModelOutputParser:
        return self._build(spec or "identity", ModelOutputParser, self.model_output_parsers, "model_output_parser", **kwargs)

    def build_action_parser(self, spec: Any, **kwargs) -> ActionParser:
        return self._build(spec, ActionParser, self.action_parsers, "action_parser", **kwargs)

    def _build(
        self,
        spec: Any,
        expected_type: type | tuple[type, ...],
        aliases: dict[str, Any],
        kind: str,
        *,
        require_factory_for_dict: bool = False,
        **kwargs,
    ):
        if isinstance(spec, expected_type):
            return spec

        spec_kwargs = {}
        factory = spec
        if isinstance(spec, dict):
            spec = dict(spec)
            factory = spec.pop("factory", None)
            if factory is None and not require_factory_for_dict:
                name = spec.pop("name", None) or spec.pop("type", None) or spec.pop("id", None)
                if not name:
                    raise TypeError(f"{kind} dict spec requires a 'name', 'type', 'id', or 'factory' field")
                factory = self._resolve_factory(name, aliases, kind)
            elif factory is None:
                raise TypeError(f"{kind} dict spec requires a callable `factory`")
            else:
                factory = self._resolve_factory(factory, aliases, kind)
            spec_kwargs = spec
        else:
            factory = self._resolve_factory(factory, aliases, kind)

        if not callable(factory):
            raise TypeError(f"Expected {kind} instance, callable, import path, or known name; got {type(spec).__name__}")

        component = call_factory(factory, {**kwargs, **spec_kwargs})
        if not isinstance(component, expected_type):
            raise TypeError(f"{kind} factory returned {type(component).__name__}, expected {_type_name(expected_type)}")
        return component

    def _resolve_factory(self, value: Any, aliases: dict[str, Any], kind: str) -> Any:
        if not isinstance(value, str):
            return value
        if value in aliases:
            value = aliases[value]
            if not isinstance(value, str):
                return value
        if ":" in value or "." in value:
            return import_from_path(value)
        available = ", ".join(sorted(aliases)) or "<none>"
        raise KeyError(f"Unknown {kind} '{value}'. Available built-ins: {available}. You can also pass an import path.")


def import_from_path(path: str) -> Any:
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, sep, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Import path must be 'module:attribute' or 'module.attribute', got {path!r}")
    module = import_module(module_name)
    return getattr(module, attr)


def call_factory(factory, kwargs: dict[str, Any]):
    signature = inspect.signature(factory)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return factory(**kwargs)

    accepted_kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return factory(**accepted_kwargs)


def _type_name(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(item.__name__ for item in expected_type)
    return expected_type.__name__


DEFAULT_AGENTIC_FACTORY = AgenticFactory()


def build_model_server(spec: Any, **kwargs) -> ModelServer:
    return DEFAULT_AGENTIC_FACTORY.build_model_server(spec, **kwargs)


def build_loop_worker(spec: Any, **kwargs) -> LoopWorker:
    return DEFAULT_AGENTIC_FACTORY.build_loop_worker(spec, **kwargs)


def build_env_manager(spec: Any, **kwargs) -> EnvManager:
    return DEFAULT_AGENTIC_FACTORY.build_env_manager(spec, **kwargs)


def build_observation_parser(spec: Any, **kwargs) -> ObservationParser:
    return DEFAULT_AGENTIC_FACTORY.build_observation_parser(spec, **kwargs)


def build_model_output_parser(spec: Any, **kwargs) -> ModelOutputParser:
    return DEFAULT_AGENTIC_FACTORY.build_model_output_parser(spec, **kwargs)


def build_action_parser(spec: Any, **kwargs) -> ActionParser:
    return DEFAULT_AGENTIC_FACTORY.build_action_parser(spec, **kwargs)

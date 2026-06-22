from __future__ import annotations

import inspect
from typing import Any


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

    def build(self, spec: Any, expected_type: type | tuple[type, ...], **kwargs):
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

        component = call_factory(factory, {**kwargs, **spec_kwargs})
        if not isinstance(component, expected_type):
            raise TypeError(f"{self.kind} factory returned {type(component).__name__}, expected {_type_name(expected_type)}")
        return component

    def names(self) -> list[str]:
        return sorted(self._items)


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


MODEL_SERVER_REGISTRY = AgenticRegistry("model_server")
LOOP_WORKER_REGISTRY = AgenticRegistry("loop_worker")
OBSERVATION_PARSER_REGISTRY = AgenticRegistry("observation_parser")
ACTION_PARSER_REGISTRY = AgenticRegistry("action_parser")
MODEL_OUTPUT_PARSER_REGISTRY = AgenticRegistry("model_output_parser")

register_model_server = MODEL_SERVER_REGISTRY.register
register_loop_worker = LOOP_WORKER_REGISTRY.register
register_observation_parser = OBSERVATION_PARSER_REGISTRY.register
register_action_parser = ACTION_PARSER_REGISTRY.register
register_model_output_parser = MODEL_OUTPUT_PARSER_REGISTRY.register

from __future__ import annotations

import itertools
import uuid
from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer


def _require_ray():
    try:
        import ray
    except ImportError as exc:
        raise ImportError("Ray model-server RPC requires `ray` to be installed.") from exc
    return ray


class RayModelServerActor:
    """Ray actor that owns one lmms-eval ModelServer instance."""

    def __init__(self, server_spec: Any, factory_components: dict[str, Any] | None = None) -> None:
        from lmms_eval.agentic.factory import DEFAULT_AGENTIC_FACTORY

        factory = DEFAULT_AGENTIC_FACTORY
        if factory_components:
            factory = factory.with_components(
                model_servers=factory_components.get("model_servers"),
                loop_workers=factory_components.get("loop_workers"),
                model_output_parsers=factory_components.get("model_output_parsers"),
                observation_parsers=factory_components.get("observation_parsers"),
                action_parsers=factory_components.get("action_parsers"),
            )
        self.server = factory.build_model_server(server_spec)

    def generate(self, request: Any) -> Any:
        return self.server.generate(request)

    def generate_batch(self, requests: list[Any]) -> list[Any]:
        return self.server.generate_batch(requests)

    def wake_up(self, *args, **kwargs) -> Any:
        method = getattr(self.server, "wake_up", None)
        return method(*args, **kwargs) if method is not None else None

    def sleep(self, *args, **kwargs) -> Any:
        method = getattr(self.server, "sleep", None)
        return method(*args, **kwargs) if method is not None else None

    def clear_kv_cache(self, *args, **kwargs) -> Any:
        method = getattr(self.server, "clear_kv_cache", None)
        return method(*args, **kwargs) if method is not None else None

    def update_weights(self, *args, **kwargs) -> Any:
        method = getattr(self.server, "update_weights", None)
        if method is None:
            raise NotImplementedError(f"{type(self.server).__name__} does not implement update_weights().")
        return method(*args, **kwargs)

    def status(self) -> dict[str, Any]:
        return {"server_type": type(self.server).__name__}


class RayModelServerLoadBalancer:
    """Small Ray load balancer for model-server actors."""

    def __init__(
        self,
        actor_names: list[str],
        namespace: str | None = None,
        sticky: bool = True,
    ) -> None:
        ray = _require_ray()
        self.namespace = namespace
        self.sticky = bool(sticky)
        self._actors = {name: ray.get_actor(name, namespace=namespace) for name in actor_names}
        self._inflight = {name: 0 for name in actor_names}
        self._request_to_actor: dict[str, str] = {}

    def acquire(self, request_id: str | None = None) -> tuple[str, Any]:
        if not self._actors:
            raise RuntimeError("No Ray model-server actors are registered.")
        if request_id and self.sticky and request_id in self._request_to_actor:
            actor_name = self._request_to_actor[request_id]
            if actor_name in self._actors:
                self._inflight[actor_name] += 1
                return actor_name, self._actors[actor_name]
            self._request_to_actor.pop(request_id, None)

        actor_name = min(self._inflight, key=lambda name: self._inflight[name])
        if request_id and self.sticky:
            self._request_to_actor[request_id] = actor_name
        self._inflight[actor_name] += 1
        return actor_name, self._actors[actor_name]

    def release(self, actor_name: str) -> None:
        if actor_name in self._inflight and self._inflight[actor_name] > 0:
            self._inflight[actor_name] -= 1

    def status(self) -> dict[str, Any]:
        return {"actors": dict(self._inflight), "sticky_sessions": len(self._request_to_actor)}


class RayActorModelServer(ModelServer):
    """ModelServer client that calls Ray actor model servers directly."""

    def __init__(
        self,
        actor_names: list[str] | str | None = None,
        actor_name: str | None = None,
        namespace: str | None = None,
        load_balancer_name: str | None = None,
        timeout_s: float | None = None,
        sticky: bool = True,
        request_id_metadata_key: str = "request_id",
        **_: Any,
    ) -> None:
        ray = _require_ray()
        self.namespace = namespace
        self.timeout_s = timeout_s
        self.sticky = bool(sticky)
        self.request_id_metadata_key = request_id_metadata_key
        self._counter = itertools.count()
        self._load_balancer = (
            ray.get_actor(load_balancer_name, namespace=namespace) if load_balancer_name is not None else None
        )

        names: list[str] = []
        if actor_name:
            names.append(actor_name)
        if isinstance(actor_names, str):
            names.extend(item.strip() for item in actor_names.split(",") if item.strip())
        elif actor_names:
            names.extend(actor_names)
        self._actors = [ray.get_actor(name, namespace=namespace) for name in names]
        if self._load_balancer is None and not self._actors:
            raise ValueError("RayActorModelServer requires actor_name, actor_names, or load_balancer_name.")

    def generate(self, request: Any) -> Any:
        request_id = self._request_id(request)
        if self._load_balancer is not None:
            actor_name, actor = self._ray_get(self._load_balancer.acquire.remote(request_id))
            try:
                return self._ray_get(actor.generate.remote(request))
            finally:
                self._load_balancer.release.remote(actor_name)

        actor = self._actors[next(self._counter) % len(self._actors)]
        return self._ray_get(actor.generate.remote(request))

    def generate_batch(self, requests: list[Any]) -> list[Any]:
        return [self.generate(request) for request in requests]

    def _ray_get(self, ref: Any) -> Any:
        ray = _require_ray()
        if self.timeout_s is None:
            return ray.get(ref)
        return ray.get(ref, timeout=self.timeout_s)

    def _request_id(self, request: Any) -> str:
        metadata = getattr(request, "metadata", {}) or {}
        if isinstance(metadata, dict):
            explicit = metadata.get(self.request_id_metadata_key)
            if explicit is not None:
                return str(explicit)
            env_id = metadata.get("env_id")
            step_idx = metadata.get("step_idx")
            agent_id = metadata.get("agent_id")
            if env_id is not None:
                return f"{env_id}:{agent_id or 'agent'}:{step_idx or 0}"
        return uuid.uuid4().hex

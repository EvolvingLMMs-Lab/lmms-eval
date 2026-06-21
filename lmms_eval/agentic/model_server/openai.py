from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.model_server.vllm import (
    _AGENTIC_ONLY_KEYS,
    _GENERATION_KEYS_TO_DROP,
    VllmModelServer,
    _apply_stop_sequences,
    _normalize_top_p,
)
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock
from lmms_eval.imports import optional_import


class OpenAIModelServer(VllmModelServer, ModelServer):
    """OpenAI-compatible HTTP model server for agentic/game loops."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        client: Any = None,
        timeout: float | int = 600,
        default_max_tokens: int = 64,
        max_parallel_rollouts: int | str = 1,
        max_concurrent_requests: int | str | None = None,
        lm: Any = None,
        doc_id: int | None = None,
        task_name: str | None = None,
        split: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        response_cache: Any = None,
        **_: Any,
    ) -> None:
        del lm, doc_id, task_name, split, request_metadata, response_cache
        self.model = model
        self.generation_kwargs = dict(generation_kwargs or {})
        self.default_max_tokens = int(default_max_tokens)
        self._max_parallel_rollouts = max(1, int(max_parallel_rollouts))
        self.max_concurrent_requests = max(1, int(max_concurrent_requests or max_parallel_rollouts))
        if client is not None:
            self.client = client
        else:
            OpenAI, has_openai = optional_import("openai", "OpenAI")
            if not has_openai:
                raise ImportError("The agentic openai model_server requires the `openai` package.")
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
                base_url=(base_url or os.getenv("OPENAI_API_BASE") or "http://127.0.0.1:8000/v1").rstrip("/"),
                timeout=timeout,
            )

    def generate(self, request: AgentInput) -> AgentOutput:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[AgentInput]) -> list[AgentOutput]:
        if not requests:
            return []
        if len(requests) == 1 or self.max_concurrent_requests <= 1:
            return [self._generate_one(request) for request in requests]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent_requests, len(requests))) as executor:
            return list(executor.map(self._generate_one, requests))

    def _generate_one(self, request: AgentInput) -> AgentOutput:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._request_to_openai_messages(request),
            **self._build_openai_params(request),
        )
        return self._response_to_agent_output(response)

    def _build_openai_params(self, request: AgentInput) -> dict[str, Any]:
        generation_kwargs = dict(self.generation_kwargs)
        generation_kwargs.update(request.generation_kwargs or {})
        params: dict[str, Any] = {
            "max_tokens": int(generation_kwargs.pop("max_tokens", generation_kwargs.pop("max_new_tokens", self.default_max_tokens))),
            "temperature": generation_kwargs.pop("temperature", 0),
            "top_p": _normalize_top_p(generation_kwargs.pop("top_p", 1.0)),
        }
        _apply_stop_sequences(params, generation_kwargs)
        for key in _AGENTIC_ONLY_KEYS | _GENERATION_KEYS_TO_DROP:
            generation_kwargs.pop(key, None)
        params.update(generation_kwargs)
        return params

    @staticmethod
    def _response_to_agent_output(response: Any) -> AgentOutput:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") if message is not None else ""
        metadata: dict[str, Any] = {"raw_response": response}
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if tool_calls:
            metadata["tool_calls"] = [_tool_call_to_dict(tool_call) for tool_call in tool_calls]
        return AgentOutput(content=[ContentBlock.text(text or "")], metadata=metadata)


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    function = getattr(tool_call, "function", None)
    arguments = getattr(function, "arguments", {}) if function is not None else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_args": arguments}
    return {
        "name": getattr(function, "name", None) if function is not None else getattr(tool_call, "name", None),
        "arguments": arguments,
        "id": getattr(tool_call, "id", None),
        "type": getattr(tool_call, "type", None),
    }

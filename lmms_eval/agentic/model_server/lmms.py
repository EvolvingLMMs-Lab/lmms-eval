from __future__ import annotations

import copy
from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock
from lmms_eval.api.instance import Instance, unwrap_generation_output


class LmmsModelServer(ModelServer):
    """Small bridge from the agentic loop to existing ``lmms.generate_until``."""

    def __init__(
        self,
        lm: Any,
        generation_kwargs: dict[str, Any] | None = None,
        doc_id: int | None = None,
        task_name: str | None = None,
        split: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        response_cache: Any = None,
        max_parallel_rollouts: int | str | None = None,
    ) -> None:
        self.lm = lm
        self.generation_kwargs = dict(generation_kwargs or {})
        self.doc_id = doc_id
        self.task_name = task_name
        self.split = split
        self.request_metadata = dict(request_metadata or {})
        self.response_cache = response_cache
        self._max_parallel_rollouts = _resolve_max_parallel_rollouts(max_parallel_rollouts, lm)

    def generate(self, request: AgentInput) -> AgentOutput:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[AgentInput]) -> list[AgentOutput]:
        if not requests:
            return []

        instances = [self._request_to_instance(request) for request in requests]
        raw_outputs = self.response_cache.execute(self.lm, "generate_until", instances) if self.response_cache is not None else self.lm.generate_until(instances)
        return [_raw_to_agent_output(raw) for raw in raw_outputs]

    def _request_to_instance(self, request: AgentInput) -> Instance:
        context, media = _split_agent_input(request)
        generation_kwargs = copy.deepcopy(self.generation_kwargs)
        generation_kwargs.update(request.generation_kwargs or {})
        doc_id, task_name, split, request_metadata = self._request_eval_context(request)
        if getattr(self.lm, "is_simple", False):
            return Instance(
                request_type="generate_until",
                arguments=(context, copy.deepcopy(generation_kwargs), lambda _doc, media=media: _media_payloads(media), doc_id, task_name, split),
                idx=0,
                metadata=request_metadata,
            )

        return Instance(
            request_type="generate_until",
            arguments=(context, _doc_to_messages(context, media), copy.deepcopy(generation_kwargs), doc_id, task_name, split),
            idx=0,
            metadata=request_metadata,
        )

    def _request_eval_context(self, request: AgentInput) -> tuple[int, str, str, dict[str, Any]]:
        context = request.metadata.get("lmms_eval") if isinstance(request.metadata, dict) else None
        context = context if isinstance(context, dict) else {}
        doc_id = context.get("doc_id", self.doc_id)
        task_name = context.get("task_name", self.task_name)
        split = context.get("split", self.split)
        request_metadata = context.get("request_metadata", self.request_metadata)
        if doc_id is None or task_name is None or split is None:
            raise ValueError("LmmsModelServer requires doc_id, task_name, and split in request.metadata['lmms_eval'] or constructor defaults")
        return int(doc_id), str(task_name), str(split), dict(request_metadata or {})


def _split_agent_input(request: AgentInput) -> tuple[str, list[ContentBlock]]:
    text_parts = []
    media = []
    for block in request.content:
        if block.type == "text" and isinstance(block.data, str):
            text_parts.append(block.data)
        elif block.type in {"image", "video", "audio"}:
            media.append(block)
    return "\n".join(text_parts), media


def _media_payloads(media: list[ContentBlock]) -> list[Any]:
    return [block.data for block in media]


def _doc_to_messages(context: str, media: list[ContentBlock]):
    def build(_doc):
        content = []
        for block in media:
            item = block.data
            media_type = block.type
            if isinstance(item, dict):
                content.append({"type": item.get("type", media_type), "url": item.get("url", item)})
            elif media_type == "video":
                content.append({"type": "video", "url": item})
            elif media_type == "audio":
                content.append({"type": "audio", "url": item})
            else:
                content.append({"type": "image", "url": item})
        if context:
            content.append({"type": "text", "text": context})
        return [{"role": "user", "content": content}]

    return build


def _raw_to_agent_output(raw: Any) -> AgentOutput:
    text, token_counts = unwrap_generation_output(raw)
    block = ContentBlock.text(text) if isinstance(text, str) else ContentBlock(type="raw", data=text)
    return AgentOutput(content=[block], metadata={"token_counts": token_counts})


def _resolve_max_parallel_rollouts(value: int | str | None, lm: Any) -> int:
    if value is not None:
        return max(1, int(value))

    for attr in ("_workers", "workers"):
        workers = getattr(lm, attr, None)
        if isinstance(workers, list | tuple) and workers:
            return len(workers)

    for attr in ("data_parallel", "data_parallel_size", "world_size"):
        candidate = getattr(lm, attr, None)
        try:
            if candidate is not None and int(candidate) > 1:
                return int(candidate)
        except (TypeError, ValueError):
            continue

    return 1

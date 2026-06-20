from __future__ import annotations

import copy
from typing import Any

from lmms_eval.agentic.model_server.base import ModelServer
from lmms_eval.agentic.types import AgentInput, AgentOutput, ContentBlock
from lmms_eval.api.instance import Instance, unwrap_generation_output


class LmmsModelServer(ModelServer):
    """Small bridge from the agentic loop to existing ``lmms.generate_until``."""

    def __init__(self, lm: Any, generation_kwargs: dict[str, Any], doc_id: int, task_name: str, split: str, request_metadata: dict[str, Any], response_cache: Any = None) -> None:
        self.lm = lm
        self.generation_kwargs = generation_kwargs
        self.doc_id = doc_id
        self.task_name = task_name
        self.split = split
        self.request_metadata = request_metadata
        self.response_cache = response_cache

    def generate(self, request: AgentInput) -> AgentOutput:
        context, media = _split_agent_input(request)
        if getattr(self.lm, "is_simple", False):
            instance = Instance(
                request_type="generate_until",
                arguments=(context, copy.deepcopy(self.generation_kwargs), lambda _doc: media, self.doc_id, self.task_name, self.split),
                idx=0,
                metadata=self.request_metadata,
            )
        else:
            instance = Instance(
                request_type="generate_until",
                arguments=(context, _doc_to_messages(context, media), copy.deepcopy(self.generation_kwargs), self.doc_id, self.task_name, self.split),
                idx=0,
                metadata=self.request_metadata,
            )

        raw = self.response_cache.execute(self.lm, "generate_until", [instance])[0] if self.response_cache is not None else self.lm.generate_until([instance])[0]
        text, token_counts = unwrap_generation_output(raw)
        block = ContentBlock.text(text) if isinstance(text, str) else ContentBlock(type="raw", data=text)
        return AgentOutput(content=[block], metadata={"token_counts": token_counts})


def _split_agent_input(request: AgentInput) -> tuple[str, list[Any]]:
    text_parts = []
    media = []
    for block in request.content:
        if block.type == "text" and isinstance(block.data, str):
            text_parts.append(block.data)
        elif block.type in {"image", "video", "audio"}:
            media.append(block.data)
    return "\n".join(text_parts), media


def _doc_to_messages(context: str, media: list[Any]):
    def build(_doc):
        content = []
        for item in media:
            if isinstance(item, dict):
                content.append({"type": item.get("type", "audio"), "url": item.get("url", item)})
            elif isinstance(item, str):
                content.append({"type": "video", "url": item})
            else:
                content.append({"type": "image", "url": item})
        if context:
            content.append({"type": "text", "text": context})
        return [{"role": "user", "content": content}]

    return build

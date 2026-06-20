from __future__ import annotations

import re
from typing import Any

from lmms_eval.agentic.parsers.base import ModelOutputParser
from lmms_eval.agentic.types import AgentOutput, ContentBlock, EnvState


class IdentityModelOutputParser(ModelOutputParser):
    """Pass model output through unchanged."""

    def parse(self, output: AgentOutput, state: EnvState, agent_id: str | None = None) -> AgentOutput:
        del state, agent_id
        return output


class QwenModelOutputParser(ModelOutputParser):
    """Normalize common Qwen chat outputs before task action parsing."""

    def __init__(self, strip_thinking: bool = True, extract_tool_calls: bool = True) -> None:
        self.strip_thinking = strip_thinking
        self.extract_tool_calls = extract_tool_calls

    def parse(self, output: AgentOutput, state: EnvState, agent_id: str | None = None) -> AgentOutput:
        del state, agent_id
        text = output.first_text() or ""
        normalized_text = _strip_thinking(text) if self.strip_thinking else text
        metadata = dict(output.metadata)
        metadata["raw_text"] = text
        metadata["normalized_text"] = normalized_text
        if self.extract_tool_calls:
            metadata["tool_calls"] = _extract_qwen_tool_calls(text)

        content = []
        replaced_text = False
        for block in output.content:
            if not replaced_text and block.type == "text" and isinstance(block.data, str):
                block_metadata = dict(block.metadata)
                block_metadata["raw_text"] = block.data
                content.append(ContentBlock.text(normalized_text, **block_metadata))
                replaced_text = True
            else:
                content.append(block)
        if not replaced_text:
            content.append(ContentBlock.text(normalized_text))
        return AgentOutput(content=content, metadata=metadata)


def _strip_thinking(text: str) -> str:
    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[-1].strip()
    if candidate.startswith("<think>") and "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[-1].strip()
    return candidate


def _extract_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    tool_calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        payload = match.group(1)
        function_match = re.search(r"<function=([^>\s]+)>(.*?)</function>", payload, flags=re.DOTALL | re.IGNORECASE)
        if function_match is None:
            continue
        params = {}
        for param_match in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", function_match.group(2), flags=re.DOTALL | re.IGNORECASE):
            params[param_match.group(1)] = param_match.group(2).strip()
        tool_calls.append({"name": function_match.group(1), "arguments": params})
    return tool_calls

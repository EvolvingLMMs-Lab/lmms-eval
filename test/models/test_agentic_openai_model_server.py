from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from lmms_eval.agentic.model_server import OpenAIModelServer
from lmms_eval.agentic.registry import MODEL_SERVER_REGISTRY, build_model_server
from lmms_eval.agentic.types import AgentInput, ContentBlock


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        text = payload["messages"][-1]["content"][0]["text"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"out:{text}", tool_calls=None))])


class _Client:
    def __init__(self):
        self.completions = _Completions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_model_server_converts_agent_input_to_openai_request():
    client = _Client()
    server = OpenAIModelServer(
        model="qwen35-vl",
        client=client,
        generation_kwargs={"max_new_tokens": 9, "temperature": 0, "top_p": 0, "do_sample": False, "until": ["\n\n"]},
    )
    frame = Image.new("RGB", (4, 4), color="red")

    output = server.generate(AgentInput(content=[ContentBlock.text("step 0"), ContentBlock(type="video", data=[frame])]))

    assert output.first_text() == "out:step 0"
    call = client.completions.calls[0]
    assert call["model"] == "qwen35-vl"
    assert call["max_tokens"] == 9
    assert call["top_p"] == 1.0
    assert call["stop"] == ["\n\n"]
    assert "do_sample" not in call
    assert "until" not in call
    assert call["messages"][0]["content"][0] == {"type": "text", "text": "step 0"}
    assert call["messages"][0]["content"][1]["type"] == "video_url"


def test_openai_model_server_batches_requests_concurrently_in_order():
    client = _Client()
    server = OpenAIModelServer(model="qwen35-vl", client=client, max_concurrent_requests=4)

    outputs = server.generate_batch([AgentInput(content=[ContentBlock.text(f"step {idx}")]) for idx in range(4)])

    assert [output.first_text() for output in outputs] == ["out:step 0", "out:step 1", "out:step 2", "out:step 3"]
    assert len(client.completions.calls) == 4


def test_agentic_registry_builds_openai_model_server():
    server = build_model_server({"name": "openai", "model": "qwen35-vl", "client": _Client(), "max_concurrent_requests": 8})

    assert isinstance(server, OpenAIModelServer)
    assert server.max_concurrent_requests == 8


def test_agentic_registry_defaults_to_openai_only():
    server = build_model_server(None, model="qwen35-vl", client=_Client())

    assert MODEL_SERVER_REGISTRY.names() == ["openai"]
    assert isinstance(server, OpenAIModelServer)


def test_openai_model_server_requires_explicit_model(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="requires model"):
        build_model_server(None, client=_Client())


def test_openai_model_server_ignores_legacy_rollout_concurrency_arg():
    server = OpenAIModelServer(model="qwen35-vl", client=_Client(), max_parallel_rollouts=3)

    assert server.max_concurrent_requests == 1

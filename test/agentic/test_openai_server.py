from __future__ import annotations

from types import SimpleNamespace

import pytest

from lmms_eval.agentic import AgentInput, ContentBlock
from lmms_eval.agentic.servers import OpenAIModelServer


class _FakeCompletions:
    def __init__(self, text="MOVE_LEFT"):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _server(**kwargs):
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    server = OpenAIModelServer(model="test-model", client=client, **kwargs)
    return server, completions


def test_generate_maps_generation_kwargs_to_openai_params():
    server, completions = _server()
    request = AgentInput(
        content=[ContentBlock.text("hello")],
        generation_kwargs={"max_new_tokens": 32, "temperature": 0.7, "top_p": 0.0, "until": ["\n\n", None], "do_sample": True, "num_beams": 2, "max_game_steps": 64},
    )

    output = server.generate(request)

    assert output.first_text() == "MOVE_LEFT"
    call = completions.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 32
    assert call["temperature"] == 0.7
    assert call["top_p"] == 1.0  # top_p=0 normalized
    assert call["stop"] == ["\n\n"]
    for leaked in ("do_sample", "num_beams", "max_game_steps", "until", "max_new_tokens"):
        assert leaked not in call


def test_server_reports_model_name_for_task_parser_selection():
    server, _ = _server()

    assert server.get_model_name() == "test-model"


def test_generate_builds_multimodal_user_message():
    server, completions = _server()
    request = AgentInput(content=[ContentBlock.text("look"), ContentBlock(type="image", data="https://example.com/frame.png")])

    server.generate(request)

    [message] = completions.calls[0]["messages"]
    assert message["role"] == "user"
    assert message["content"][0] == {"type": "text", "text": "look"}
    assert message["content"][1] == {"type": "image_url", "image_url": {"url": "https://example.com/frame.png"}}


def test_generate_prepends_conversation_history():
    server, completions = _server()
    history = [
        {"role": "user", "content": [ContentBlock.text("earlier obs")]},
        {"role": "assistant", "content": "ATTACK"},
    ]
    request = AgentInput(content=[ContentBlock.text("now")], metadata={"conversation_history": history})

    server.generate(request)

    messages = completions.calls[0]["messages"]
    assert len(messages) == 3
    assert messages[0]["content"] == [{"type": "text", "text": "earlier obs"}]
    assert messages[1] == {"role": "assistant", "content": "ATTACK"}
    assert messages[2]["content"] == [{"type": "text", "text": "now"}]


def test_enable_thinking_rides_extra_body_chat_template_kwargs():
    server, completions = _server(enable_thinking=False)
    server.generate(AgentInput(content=[ContentBlock.text("x")]))

    assert completions.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_requires_model_name(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="model="):
        OpenAIModelServer(model=None, client=SimpleNamespace())


def test_rejects_non_agent_input():
    server, _ = _server()
    with pytest.raises(TypeError, match="AgentInput"):
        server.generate("plain string")

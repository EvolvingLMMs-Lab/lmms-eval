"""Focused tests for empty choices, missing messages, and blank content."""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from lmms_eval.api.chat_response import (
    EMPTY_CHAT_RESPONSE,
    require_chat_message_content,
    require_chat_message_text,
    require_choice_message_text,
)
from lmms_eval.llm_judge.protocol import Request, ServerConfig
from lmms_eval.llm_judge.providers.async_openai import AsyncOpenAIProvider
from lmms_eval.llm_judge.providers.openai import OpenAIProvider


def _response(*, choices):
    return SimpleNamespace(choices=choices, model="gpt-test", usage=None)


def _message(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


@pytest.mark.parametrize(
    "response",
    [
        _response(choices=[]),
        _response(choices=None),
        {},
        {"choices": []},
        _response(choices=[SimpleNamespace(message=None)]),
        {"choices": [{"message": None}]},
        _response(choices=[_message(None)]),
        {"choices": [{"message": {"content": None}}]},
        _response(choices=[_message("")]),
        _response(choices=[_message("   \n")]),
        _response(choices=[_message([])]),
    ],
)
def test_missing_or_empty_content_raises_before_strip(response):
    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        require_chat_message_text(response).strip()
    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        require_chat_message_content(response)


def test_text_and_parts_pass_through_without_calling_strip_on_none():
    assert require_chat_message_text(_response(choices=[_message("  hello  ")])) == "  hello  "
    assert require_chat_message_text(_response(choices=[_message("  hello  ")])).strip() == "hello"
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert require_chat_message_content(_response(choices=[_message(parts)])) == parts
    assert require_chat_message_text(_response(choices=[_message(parts)])) == "ab"


def _provider():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = ServerConfig(model_name="gpt-4o-mini", num_retries=1, retry_delay=0, max_tokens=8, temperature=0)
    provider.api_key = "test"
    provider.base_url = "https://example.invalid/v1"
    provider.use_client = True
    provider.client = mock.Mock()
    return provider


def _request():
    return Request(messages=[{"role": "user", "content": "ping"}], config=ServerConfig(model_name="gpt-4o-mini", num_retries=1, retry_delay=0, max_tokens=8, temperature=0))


@pytest.mark.parametrize(
    "payload",
    [
        _response(choices=[]),
        _response(choices=[SimpleNamespace(message=None)]),
        _response(choices=[_message(None)]),
        _response(choices=[_message("  ")]),
    ],
)
def test_openai_judge_raises_for_invalid_response(payload):
    provider = _provider()
    provider.client.chat.completions.create.return_value = payload
    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        provider.evaluate(_request())
    assert provider.client.chat.completions.create.call_count == 1


def test_openai_judge_strips_valid_content():
    provider = _provider()
    provider.client.chat.completions.create.return_value = _response(choices=[_message("  yes  ")])
    result = provider.evaluate(_request())
    assert result.content == "yes"


def test_openai_judge_dict_payload_rejects_missing_content():
    provider = _provider()
    provider.use_client = False
    provider._make_request = mock.Mock(return_value={"choices": [{"message": {"content": None}}], "model": "gpt-test"})
    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        provider.evaluate(_request())


def test_each_choice_rejects_blank_content_before_strip():
    choices = [{"message": {"content": "A"}}, {"message": {"content": None}}]
    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        [require_choice_message_text(choice).strip() for choice in choices]
    assert require_choice_message_text({"message": {"content": "  B  "}}).strip() == "B"


def _async_provider():
    provider = AsyncOpenAIProvider.__new__(AsyncOpenAIProvider)
    provider.config = ServerConfig(model_name="gpt-4o-mini", num_retries=1, retry_delay=0, max_tokens=8, temperature=0)
    provider.api_key = "test"
    provider.use_async_client = True
    provider.async_client = mock.Mock()
    provider.semaphore = asyncio.Semaphore(1)
    return provider


@pytest.mark.parametrize(
    "payload",
    [
        _response(choices=[]),
        _response(choices=[SimpleNamespace(message=None)]),
        _response(choices=[_message(None)]),
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_async_openai_judge_raises_for_invalid_response(payload):
    provider = _async_provider()
    provider.async_client.chat.completions.create = mock.AsyncMock(return_value=payload)

    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        asyncio.run(provider.evaluate_async(_request()))

    assert provider.async_client.chat.completions.create.await_count == 1


def test_async_openai_judge_strips_valid_content():
    provider = _async_provider()
    provider.async_client.chat.completions.create = mock.AsyncMock(return_value=_response(choices=[_message("  yes  ")]))
    result = asyncio.run(provider.evaluate_async(_request()))
    assert result.content == "yes"


def _chat_turn(*, finish_reason, content, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def test_async_openai_chat_allows_null_content_on_tool_call_turn():
    from lmms_eval.models.chat.async_openai import _turn_text

    turn = _chat_turn(finish_reason="tool_calls", content=None, tool_calls=[SimpleNamespace(id="call_1")])
    assert _turn_text(turn) == ""


def test_async_openai_chat_still_rejects_blank_final_turn():
    from lmms_eval.models.chat.async_openai import _turn_text

    with pytest.raises(ValueError, match=EMPTY_CHAT_RESPONSE):
        _turn_text(_chat_turn(finish_reason="stop", content=None))

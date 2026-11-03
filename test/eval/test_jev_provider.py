"""HTTP-level tests for the TypeSafe Jev judge provider.

Jev is a System One model: it returns noul, choice, and score answers, not
generated text. These tests mock ``requests.post`` and check the request
body, the canonical string tasks already parse, and failure handling.
"""

import asyncio
import json
from unittest import mock

import pytest
import requests

from lmms_eval.llm_judge.factory import ProviderFactory
from lmms_eval.llm_judge.protocol import Request, ServerConfig
from lmms_eval.llm_judge.providers.async_jev import AsyncJevProvider
from lmms_eval.llm_judge.providers.jev import (
    JEV_API_URL,
    JEV_EMPTY_RESPONSE,
    JevAPIError,
    JevProvider,
    JevTextOnlyError,
    JevUnsupportedPromptError,
)


def _http(status, body, headers=None):
    response = mock.Mock()
    response.status_code = status
    response.headers = headers or {}
    response.text = json.dumps(body)
    response.json.return_value = body
    return response


def _usage(body):
    body.setdefault("usage", {"input_tokens": 12, "output_tokens": 3})
    return body


def _noul(probability, model="jev-1.14.0"):
    return _usage({"model": model, "answers": {"judgment": {"type": "noul", "noul": probability}}})


def _score_body(scores, model="jev-1.14.0"):
    answers = {}
    for key, score in scores.items():
        answers[key] = {
            "type": "score",
            "score": score,
            "legend": {"0": "low", "1": "high"},
            "probabilities": {"0": 0.2, "1": 0.8},
            "confidence": 0.7,
        }
    return _usage({"model": model, "answers": answers})


def _choice_body(option="billing", model="jev-1.14.0"):
    return _usage(
        {
            "model": model,
            "answers": {
                "judgment": {
                    "type": "choice",
                    "choice": option,
                    "probabilities": {"billing": 0.88, "technical": 0.12},
                    "confidence": 0.81,
                }
            },
        }
    )


def _config(**kwargs) -> ServerConfig:
    kwargs.setdefault("model_name", "jev-latest")
    kwargs.setdefault("num_retries", 1)
    kwargs.setdefault("retry_delay", 0)
    kwargs.setdefault("timeout", 3)
    return ServerConfig(**kwargs)


def _provider(monkeypatch, **config_kwargs) -> JevProvider:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return JevProvider(_config(**config_kwargs))


def _posted(post):
    return post.call_args.kwargs["json"]


class TestJevRegistration:
    def test_factory_names(self):
        assert ProviderFactory._provider_classes["jev"] is JevProvider
        assert ProviderFactory._provider_classes["typesafe"] is JevProvider
        assert ProviderFactory._provider_classes["async_jev"] is AsyncJevProvider
        assert ProviderFactory._provider_classes["async_typesafe"] is AsyncJevProvider

    def test_environment_selects_jev(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
        monkeypatch.setenv("API_TYPE", "typesafe")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        provider = ProviderFactory.create_provider()

        assert isinstance(provider, JevProvider)
        assert provider.is_available() is True
        assert provider.api_key == "test-key"
        assert provider.config.model_name == "jev-latest"

    def test_missing_key_ignores_openai_key(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        provider = JevProvider()

        assert provider.is_available() is False
        assert provider.config.model_name == "jev-latest"
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            provider.evaluate(Request(messages=[{"role": "user", "content": "Is 2+2=4? Answer Yes or No."}]))


class TestJevNoul:
    def test_binary_request_shape_and_yes_adapter(self, monkeypatch):
        provider = _provider(monkeypatch)
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.81))) as post:
            result = provider.evaluate_binary("What is 2+2?", "4", "4", output_format="yes/no")

        payload = _posted(post)
        assert post.call_args.args[0] == JEV_API_URL
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert post.call_args.kwargs["timeout"] == 3
        assert payload["model"] == "jev-latest"
        assert payload["state"]["question"] == "What is 2+2?"
        assert payload["state"]["reference_answer"] == "4"
        assert payload["state"]["prediction"] == "4"
        assert "judge_prompt" in payload["state"]
        question = payload["questions"]["judgment"]
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}
        assert result["result"] is True
        assert result["probability"] == 0.81
        assert result["threshold"] == 0.5
        assert result["raw_response"] == "Yes"
        assert result["success"] is True

    def test_threshold_boundary_and_zero_one_adapter(self, monkeypatch):
        provider = _provider(monkeypatch, noul_threshold=0.5)
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.5))):
            positive = provider.evaluate_binary("q", "a", "p", output_format="0/1")
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.49))):
            negative = provider.evaluate_binary("q", "a", "p", output_format="0/1")

        assert positive["result"] == 1
        assert positive["raw_response"] == "1"
        assert negative["result"] == 0
        assert negative["raw_response"] == "0"
        assert negative["probability"] == 0.49

    def test_env_threshold_overrides_default(self, monkeypatch):
        monkeypatch.setenv("JEV_NOUL_THRESHOLD", "0.9")
        provider = _provider(monkeypatch)
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.85))) as post:
            result = provider.evaluate(Request(messages=[{"role": "user", "content": 'Return only "Yes" or "No".'}], config=_config()))

        assert result.content == "No"
        assert result.parsed_result["probability"] == 0.85
        assert result.parsed_result["threshold"] == 0.9
        assert result.parsed_result["decision"] is False
        assert result.usage["input_tokens"] == 12
        assert result.usage["prompt_tokens"] == 12
        assert result.model_used == "jev-1.14.0"
        assert _posted(post)["model"] == "jev-latest"

    def test_prompt_styles(self, monkeypatch):
        provider = _provider(monkeypatch)
        cases = [
            ('Respond with "True" or "False".', 0.2, "False"),
            ("Correct_or_not: compare the final answer.", 0.95, "Correct"),
            ("Output <label>True</label> or <label>False</label>.", 0.1, "<label>False</label>"),
        ]
        for prompt, probability, expected in cases:
            with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(probability))):
                result = provider.evaluate(Request(messages=[{"role": "user", "content": prompt}], config=_config()))
            assert result.content == expected

    def test_non_jev_model_name_uses_jev_latest(self, monkeypatch):
        provider = _provider(monkeypatch, model_name="gpt-4o-2024-11-20")
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.99))) as post:
            provider.evaluate_binary("q", "a", "p", output_format="yes/no")
        assert _posted(post)["model"] == "jev-latest"

    def test_versioned_model_and_typesafe_model_env(self, monkeypatch):
        provider = _provider(monkeypatch, model_name="jev-1.14.0")
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.6))) as post:
            provider.evaluate_binary("q", "a", "p", output_format="0/1")
        assert _posted(post)["model"] == "jev-1.14.0"

        monkeypatch.setenv("TYPESAFE_MODEL", "jev-1.13.0")
        provider = _provider(monkeypatch, model_name="gpt-4o-mini")
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.6))) as post:
            provider.evaluate_binary("q", "a", "p", output_format="0/1")
        assert _posted(post)["model"] == "jev-1.13.0"


class TestJevScoreAndChoice:
    def test_score_maps_level_index_onto_task_scale(self, monkeypatch):
        provider = _provider(monkeypatch, judge_type="score", score_range=(0.0, 1.0))
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _score_body({"judgment": 0.4}))) as post:
            result = provider.evaluate(Request(messages=[{"role": "user", "content": "Score the answer."}], config=provider.config))

        criteria = _posted(post)["questions"]["judgment"]["criteria"]
        assert _posted(post)["questions"]["judgment"]["type"] == "score"
        assert isinstance(criteria, list)
        assert 2 <= len(criteria) <= 10
        assert result.content == "0.4"
        assert result.parsed_result["score"] == 0.4
        assert result.parsed_result["confidence"] == 0.7

    def test_unit_interval_uses_two_endpoints(self, monkeypatch):
        provider = _provider(monkeypatch)
        prompt = "The correctness score is 0.0 (totally wrong), 0.5, or 1.0 (totally right)."
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _score_body({"judgment": 1.0}))) as post:
            result = provider.evaluate(Request(messages=[{"role": "user", "content": prompt}], config=_config()))

        criteria = _posted(post)["questions"]["judgment"]["criteria"]
        assert len(criteria) == 2
        assert result.content == "1"

    def test_integer_scale_maps_endpoints(self, monkeypatch):
        provider = _provider(monkeypatch, judge_type="score", score_range=(1, 10))
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _score_body({"judgment": 9}))) as post:
            top = provider.evaluate(Request(messages=[{"role": "user", "content": "Rate it."}], config=provider.config))
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _score_body({"judgment": 0}))):
            bottom = provider.evaluate(Request(messages=[{"role": "user", "content": "Rate it."}], config=provider.config))

        assert len(_posted(post)["questions"]["judgment"]["criteria"]) == 10
        assert top.content == "10"
        assert bottom.content == "1"

    def test_rubric_levels_are_sent_as_score_criteria(self, monkeypatch):
        provider = _provider(monkeypatch)
        rubric = {"accuracy": ["wrong", "partial", "right"]}
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _score_body({"accuracy": 2}))) as post:
            result = provider.evaluate_with_rubric("2+2?", "4", rubric)

        question = _posted(post)["questions"]["accuracy"]
        assert question["type"] == "score"
        assert question["criteria"] == ["wrong", "partial", "right"]
        assert result["scores"] == {"accuracy": 2.0}
        assert result["success"] is True

    def test_choice_request_and_adapter(self, monkeypatch):
        provider = _provider(monkeypatch)
        request = Request(
            messages=[{"role": "user", "content": "Which team?"}],
            options=["billing", "technical"],
            config=_config(judge_type="choice"),
        )
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _choice_body())) as post:
            result = provider.evaluate(request)

        question = _posted(post)["questions"]["judgment"]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"billing", "technical"}
        assert result.content == "billing"
        assert result.parsed_result["probabilities"]["billing"] == 0.88
        assert result.parsed_result["confidence"] == 0.81

    def test_comparative_returns_two_scores(self, monkeypatch):
        provider = _provider(monkeypatch)
        body = _score_body({"response_1": 0, "response_2": 9})
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, body)) as post:
            result = provider.evaluate_comparative("Question?", "left", "right", score_range=(1, 10))

        questions = _posted(post)["questions"]
        assert set(questions) == {"response_1", "response_2"}
        assert questions["response_1"]["type"] == "score"
        assert result["scores"] == (1.0, 10.0)
        assert result["raw_response"] == "1 10"

    def test_too_many_options_fails_before_http(self, monkeypatch):
        provider = _provider(monkeypatch)
        request = Request(messages=[{"role": "user", "content": "Pick"}], options=[str(i) for i in range(256)], config=_config(judge_type="choice"))
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post") as post:
            with pytest.raises(ValueError, match="255"):
                provider.evaluate(request)
        post.assert_not_called()


class TestJevFailures:
    def test_images_fail_before_http(self, monkeypatch):
        provider = _provider(monkeypatch)
        request = Request(messages=[{"role": "user", "content": "Describe"}], images=[b"jpeg"], config=_config())
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post") as post:
            with pytest.raises(JevTextOnlyError, match="text-only"):
                provider.evaluate(request)
        post.assert_not_called()

    def test_image_parts_and_data_urls_fail(self, monkeypatch):
        provider = _provider(monkeypatch)
        messages = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaaa"}}]}]
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post") as post:
            with pytest.raises(JevTextOnlyError, match="text-only"):
                provider.evaluate(Request(messages=messages, config=_config()))
        post.assert_not_called()

    def test_extraction_prompt_is_rejected(self, monkeypatch):
        provider = _provider(monkeypatch)
        prompt = "Demo\n\nQuestion: 1+1\n\nModel response: two\n\nExtracted answer: "
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post") as post:
            with pytest.raises(JevUnsupportedPromptError, match="does not generate"):
                provider.evaluate(Request(messages=[{"role": "user", "content": prompt}], config=_config()))
        post.assert_not_called()

    def test_client_error_is_not_retried(self, monkeypatch):
        provider = _provider(monkeypatch, num_retries=4)
        body = {"detail": "criteria must be an object"}
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(422, body)) as post:
            with mock.patch("lmms_eval.llm_judge.providers.jev.time.sleep") as sleep:
                with pytest.raises(JevAPIError, match="422") as caught:
                    provider.evaluate_binary("q", "a", "p", output_format="yes/no")
        assert post.call_count == 1
        sleep.assert_not_called()
        assert "test-key" not in str(caught.value)

    def test_retries_429_and_5xx_with_backoff(self, monkeypatch):
        provider = _provider(monkeypatch, num_retries=4, retry_delay=1)
        responses = [
            _http(429, {"error": "slow down"}, headers={"Retry-After": "7"}),
            _http(500, {"error": "boom"}),
            _http(529, {"error": "overloaded"}),
            _http(200, _noul(0.7)),
        ]
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", side_effect=responses) as post:
            with mock.patch("lmms_eval.llm_judge.providers.jev.time.sleep") as sleep:
                result = provider.evaluate_binary("q", "a", "p", output_format="yes/no")

        assert result["raw_response"] == "Yes"
        assert post.call_count == 4
        assert [call.args[0] for call in sleep.call_args_list] == [7.0, 2.0, 4.0]

    def test_timeout_then_success(self, monkeypatch):
        provider = _provider(monkeypatch, num_retries=2, retry_delay=0)
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", side_effect=[requests.Timeout("timed out"), _http(200, _noul(1))]) as post:
            result = provider.evaluate_binary("q", "a", "p", output_format="0/1")
        assert result["result"] == 1
        assert post.call_count == 2

    def test_timeout_exhaustion_raises(self, monkeypatch):
        provider = _provider(monkeypatch, num_retries=2, retry_delay=0)
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(requests.Timeout, match="timed out"):
                provider.evaluate_binary("q", "a", "p", output_format="yes/no")

    def test_empty_answer_raises_without_stripping_none(self, monkeypatch):
        provider = _provider(monkeypatch, num_retries=1)
        empty_cases = [
            {"model": "jev-1.14.0", "answers": {}, "usage": {}},
            {"model": "jev-1.14.0", "answers": {"judgment": {"type": "noul"}}, "usage": {}},
            {"model": "jev-1.14.0", "answers": {"judgment": {"type": "noul", "noul": None}}, "usage": {}},
            {"model": "jev-1.14.0", "answers": {"judgment": {"type": "choice", "choice": ""}}, "usage": {}},
        ]
        for body in empty_cases:
            with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, body)) as post:
                with pytest.raises(ValueError, match=JEV_EMPTY_RESPONSE):
                    provider.evaluate_binary("q", "a", "p", output_format="yes/no")
            assert post.call_count == 1

    def test_one_failed_judgment_does_not_break_the_next(self, monkeypatch):
        provider = _provider(monkeypatch, num_retries=1)
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(401, {"error": "unauthorized"})):
            with pytest.raises(JevAPIError, match="401"):
                provider.evaluate_binary("q", "a", "bad", output_format="yes/no")
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.99))):
            result = provider.evaluate_binary("q", "a", "good", output_format="yes/no")
        assert result["result"] is True


class TestAsyncJev:
    def test_async_evaluate_uses_same_payload(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
        provider = AsyncJevProvider(_config(max_concurrent=2))
        request = Request(messages=[{"role": "user", "content": 'Return only "Yes" or "No".'}], config=_config())
        with mock.patch("lmms_eval.llm_judge.providers.jev.requests.post", return_value=_http(200, _noul(0.66))) as post:
            result = asyncio.run(provider.evaluate_async(request))

        assert result.content == "Yes"
        assert result.parsed_result["probability"] == 0.66
        assert _posted(post)["questions"]["judgment"]["type"] == "noul"
        assert provider.semaphore._value == 2

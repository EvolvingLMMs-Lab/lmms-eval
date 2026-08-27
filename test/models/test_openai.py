from __future__ import annotations

import unittest
from types import SimpleNamespace

from lmms_eval.models.chat.openai import OpenAICompatible as ChatOpenAICompatible
from lmms_eval.models.simple.openai import OpenAICompatible as SimpleOpenAICompatible


def _fake_response(content: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    return SimpleNamespace(choices=[choice], usage=None)


class _CaptureCompletions:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def create(self, **payload):
        self.payloads.append(payload)
        return _fake_response()


def _request(*args) -> SimpleNamespace:
    return SimpleNamespace(args=args)


def _chat_request(gen_kwargs: dict) -> SimpleNamespace:
    return _request(
        "",
        lambda _doc: [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Respond to this"}],
            }
        ],
        gen_kwargs,
        0,
        "demo",
        "test",
    )


def _configure_openai_model(model, completions: _CaptureCompletions, *, model_version: str = "gpt-4o") -> None:
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model.model_version = model_version
    model.max_retries = 1
    model.retry_backoff_s = 0
    model.num_concurrent = 1
    model.adaptive_concurrency = False
    model.adaptive_config = SimpleNamespace(max_concurrency=1)
    model.prefix_aware_queue = False
    model.prefix_hash_chars = 256
    model.max_frames_num = 1
    model.video_fps = None
    model.pass_video_url = False
    model.enable_thinking_kwarg = None
    model._rank = 0
    model.task_dict = {"demo": {"test": [{"id": 0}]}}


class TestOpenAICompatibleGenerationParameters(unittest.TestCase):
    def test_simple_backend_preserves_requested_max_new_tokens(self):
        completions = _CaptureCompletions()
        model = SimpleOpenAICompatible.__new__(SimpleOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until(
            [
                _request(
                    "Describe the image",
                    {"max_new_tokens": 8192, "temperature": 0},
                    lambda _doc: None,
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(completions.payloads[0]["max_tokens"], 8192)

    def test_chat_backend_preserves_requested_max_new_tokens(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Describe this"}],
                        }
                    ],
                    {"max_new_tokens": 32768, "temperature": 0},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(completions.payloads[0]["max_tokens"], 32768)

    def test_chat_reasoning_models_use_requested_completion_tokens(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions, model_version="gpt-5")

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Reason carefully"}],
                        }
                    ],
                    {"max_new_tokens": 32768, "temperature": 0.7},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertNotIn("max_tokens", completions.payloads[0])
        self.assertEqual(completions.payloads[0]["max_completion_tokens"], 32768)

    def test_chat_backend_preserves_per_request_sampling_parameters(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)
        generation_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.4,
            "top_p": 0.8,
            "seed": 42,
            "presence_penalty": 0.1,
            "frequency_penalty": -0.2,
        }
        original_generation_kwargs = dict(generation_kwargs)

        model.generate_until([_chat_request(generation_kwargs)])

        payload = completions.payloads[0]
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["presence_penalty"], 0.1)
        self.assertEqual(payload["frequency_penalty"], -0.2)
        self.assertEqual(generation_kwargs, original_generation_kwargs)

    def test_chat_backend_accepts_one_response_choice(self):
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until([_chat_request({"n": 1})])

        self.assertEqual(completions.payloads[0]["n"], 1)

    def test_chat_backend_rejects_unsupported_response_choice_counts(self):
        for invalid_n in (0, 2, -1, 1.0, True, "1"):
            with self.subTest(n=invalid_n):
                completions = _CaptureCompletions()
                model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
                _configure_openai_model(model, completions)

                with self.assertRaisesRegex(ValueError, "n.*integer 1"):
                    model.generate_until([_chat_request({"n": invalid_n})])

                self.assertEqual(completions.payloads, [])


if __name__ == "__main__":
    unittest.main()

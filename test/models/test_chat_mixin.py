"""Regression tests for per-request handling in the shared chat model loop."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from lmms_eval.api.instance import Instance
from lmms_eval.models.chat._chat_base import ChatMixin
from lmms_eval.protocol import ChatMessages


class _Progress:
    def __init__(self, total: int, **_kwargs) -> None:
        self.total = total
        self.n = 0
        self.closed = False

    def update(self, count: int) -> None:
        self.n += count

    def close(self) -> None:
        self.closed = True


class _CacheHook:
    def __init__(self) -> None:
        self.entries = []

    def add_partial(self, method, key, answer) -> None:
        self.entries.append((method, key, answer))


class _Model(ChatMixin):
    def __init__(self, batch_size: int, task_dict: dict) -> None:
        self.batch_size = batch_size
        self.rank = 0
        self.task_dict = task_dict
        self.cache_hook = _CacheHook()
        self.calls = []
        self.fail_on = set()
        self.mutate_kwargs = False

    def _infer_one(self, chat_messages: ChatMessages, gen_kwargs: dict) -> str:
        text = chat_messages.messages[0].content[0].text
        self.calls.append((text, dict(gen_kwargs), gen_kwargs))
        if self.mutate_kwargs:
            gen_kwargs["inference_only"] = text
        if text in self.fail_on:
            raise RuntimeError(text)
        return text


def _request(context: str, callback, kwargs: dict, doc_id: int, task: str, split: str) -> Instance:
    return Instance(
        request_type="generate_until",
        arguments=(context, callback, kwargs, doc_id, task, split),
        idx=doc_id,
        metadata={"task": task, "doc_id": doc_id, "repeats": 1},
    )


def _callback(prefix: str):
    def doc_to_messages(doc: dict) -> list:
        return [{"role": "user", "content": [{"type": "text", "text": f"{prefix}:{doc['marker']}"}]}]

    return doc_to_messages


def _run(model: _Model, requests: list[Instance], progress: list | None = None):
    if progress is None:
        progress = []

    def make_progress(**kwargs):
        bar = _Progress(**kwargs)
        progress.append(bar)
        return bar

    with patch("lmms_eval.models.chat._chat_base.tqdm", side_effect=make_progress):
        result = model.generate_until(requests)
    return result, progress[0]


class TestChatMixinGenerateUntil(unittest.TestCase):
    def test_all_requests_keep_their_owners_and_original_order(self) -> None:
        """Batches and uneven tails must consume every full request tuple."""
        task_dict = {
            "alpha": {"test": [{"marker": "a0"}, {"marker": "a1"}], "dev": [{"marker": "ad0"}]},
            "beta": {"test": [{"marker": "b0"}, {"marker": "b1"}]},
        }
        shared_kwargs = {"temperature": 0.2, "max_new_tokens": 7}
        requests = [
            _request("z-long", _callback("A"), shared_kwargs, 0, "alpha", "test"),
            _request("a", _callback("B"), shared_kwargs, 0, "beta", "test"),
            _request("m-middle", _callback("C"), shared_kwargs, 1, "alpha", "test"),
            _request("b", _callback("D"), shared_kwargs, 0, "alpha", "dev"),
            _request("y-longer", _callback("E"), shared_kwargs, 1, "beta", "test"),
        ]
        expected = ["A:a0", "B:b0", "C:a1", "D:ad0", "E:b1"]

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                model = _Model(batch_size, task_dict)
                result, bar = _run(model, requests)
                self.assertEqual(result, expected)
                self.assertCountEqual([call[0] for call in model.calls], expected)
                self.assertEqual(len(model.cache_hook.entries), len(requests))
                self.assertCountEqual(
                    [(key[0], answer) for method, key, answer in model.cache_hook.entries if method == "generate_until"],
                    list(zip([request.args[0] for request in requests], expected)),
                )
                self.assertTrue(all(call[1] == shared_kwargs for call in model.calls))
                self.assertEqual((bar.n, bar.total, bar.closed), (len(requests), len(requests), True))
                self.assertEqual(shared_kwargs, {"temperature": 0.2, "max_new_tokens": 7})

    def test_kwargs_are_independent_copies_and_input_is_unchanged(self) -> None:
        kwargs = {"temperature": 0.2}
        task_dict = {"task": {"test": [{"marker": "one"}, {"marker": "two"}]}}
        callback = _callback("message")
        requests = [_request(str(i), callback, kwargs, i, "task", "test") for i in range(2)]
        model = _Model(2, task_dict)
        model.mutate_kwargs = True

        result, _ = _run(model, requests)

        self.assertEqual(result, ["message:one", "message:two"])
        self.assertEqual([call[1] for call in model.calls], [kwargs, kwargs])
        self.assertIsNot(model.calls[0][2], model.calls[1][2])
        self.assertIsNot(model.calls[0][2], kwargs)
        self.assertIsNot(model.calls[1][2], kwargs)
        self.assertEqual(kwargs, {"temperature": 0.2})

    def test_interleaved_kwargs_groups_keep_per_request_settings(self) -> None:
        task_dict = {"task": {"test": [{"marker": str(i)} for i in range(4)]}}
        callback = _callback("message")
        normal = {"temperature": 0.1}
        strict = {"temperature": 0.9, "raise_on_error": True}
        requests = [_request(context, callback, kwargs, doc_id, "task", "test") for doc_id, (context, kwargs) in enumerate([("z", normal), ("a", strict), ("y", normal), ("b", strict)])]
        model = _Model(2, task_dict)
        model.fail_on = {"message:2"}

        result, bar = _run(model, requests)

        self.assertEqual(result, ["message:0", "message:1", "", "message:3"])
        self.assertEqual(
            {(text, kwargs["temperature"]) for text, kwargs, _ in model.calls},
            {("message:0", 0.1), ("message:1", 0.9), ("message:2", 0.1), ("message:3", 0.9)},
        )
        self.assertCountEqual(
            [(key[0], key[1], answer) for _, key, answer in model.cache_hook.entries],
            [
                ("z", {"temperature": 0.1}, "message:0"),
                ("a", {"temperature": 0.9}, "message:1"),
                ("y", {"temperature": 0.1}, ""),
                ("b", {"temperature": 0.9}, "message:3"),
            ],
        )
        self.assertEqual((bar.n, bar.total, bar.closed), (4, 4, True))
        self.assertEqual(strict, {"temperature": 0.9, "raise_on_error": True})

    def test_inference_failure_continues_or_propagates_per_request(self) -> None:
        task_dict = {"task": {"test": [{"marker": "one"}, {"marker": "bad"}, {"marker": "three"}]}}
        callback = _callback("message")
        requests = [_request(str(i), callback, {}, i, "task", "test") for i in range(3)]
        model = _Model(4, task_dict)
        model.fail_on = {"message:bad"}

        result, bar = _run(model, requests)
        self.assertEqual(result, ["message:one", "", "message:three"])
        self.assertEqual([entry[2] for entry in model.cache_hook.entries], result)
        self.assertEqual((bar.n, bar.closed), (3, True))

        requests = [_request(str(i), callback, {"raise_on_error": True}, i, "task", "test") for i in range(3)]
        model = _Model(4, task_dict)
        model.fail_on = {"message:bad"}
        progress = []
        with self.assertRaisesRegex(RuntimeError, "message:bad"):
            _run(model, requests, progress)
        self.assertEqual([call[0] for call in model.calls], ["message:one", "message:bad"])
        self.assertEqual(requests[1].args[2], {"raise_on_error": True})
        self.assertEqual((progress[0].n, progress[0].closed), (1, True))

    def test_empty_requests(self) -> None:
        model = _Model(4, {})
        result, bar = _run(model, [])
        self.assertEqual(result, [])
        self.assertEqual(model.calls, [])
        self.assertEqual(model.cache_hook.entries, [])
        self.assertEqual((bar.n, bar.total, bar.closed), (0, 0, True))

    def test_message_construction_errors_stay_outside_inference_fallback(self) -> None:
        def broken_callback(_doc: dict) -> list:
            raise ValueError("invalid document")

        model = _Model(2, {"task": {"test": [{"marker": "bad"}]}})
        request = _request("ctx", broken_callback, {}, 0, "task", "test")
        progress = []
        with self.assertRaisesRegex(ValueError, "invalid document"):
            _run(model, [request], progress)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.cache_hook.entries, [])
        self.assertEqual((progress[0].n, progress[0].closed), (0, True))

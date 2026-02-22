import json
import os
import shutil
import tempfile
import unittest

from lmms_eval.api.instance import Instance
from lmms_eval.caching.response_cache import ResponseCache
from lmms_eval.evaluator import _run_generate_until_agentic


def _terminal_doc_to_text(_doc, previous_output, round_idx, previous_round_info):
    return (
        None,
        None,
        True,
        previous_output,
        {
            "state": {"round": round_idx},
            "tool_calls": 0,
            "valid_tool_calls": 0,
            "invalid_steps": 0,
            "prev": previous_round_info,
        },
    )


class _FakeSimpleLM:
    is_simple = True

    def __init__(self):
        self.calls = 0
        self.task_dict = {"agentic_task": {"test": [{"id": 0}]}}

    def generate_until(self, requests):
        self.calls += len(requests)
        return [f"raw::{req.args[0]}" for req in requests]


def _make_agentic_request(prompt: str) -> Instance:
    return Instance(
        request_type="generate_until_agentic",
        arguments=(
            prompt,
            {"temperature": 0, "max_agentic_steps": 2},
            lambda _doc: [],
            _terminal_doc_to_text,
            0,
            "agentic_task",
            "test",
        ),
        idx=0,
        metadata={"task": "agentic_task", "doc_id": 0, "repeats": 1},
    )


class TestAgenticResponseCache(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "rank0.db")
        self.audit_path = os.path.join(self.tmpdir, "rank0.jsonl")
        self.cache = ResponseCache(self.db_path, self.audit_path, model_fingerprint="agentic-test")

    def tearDown(self):
        self.cache.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_agentic_path_uses_response_cache(self):
        lm = _FakeSimpleLM()
        req = _make_agentic_request("prompt_a")

        out1 = _run_generate_until_agentic(lm, [req], response_cache=self.cache)
        self.assertEqual(lm.calls, 1)

        out2 = _run_generate_until_agentic(lm, [req], response_cache=self.cache)
        self.assertEqual(lm.calls, 1)
        self.assertEqual(out1, out2)

    def test_agentic_same_doc_different_prompt_not_collide(self):
        lm = _FakeSimpleLM()
        req_a = _make_agentic_request("prompt_a")
        req_b = _make_agentic_request("prompt_b")

        _run_generate_until_agentic(lm, [req_a], response_cache=self.cache)
        self.assertEqual(lm.calls, 1)

        out_b = _run_generate_until_agentic(lm, [req_b], response_cache=self.cache)
        self.assertEqual(lm.calls, 2)

        payload_b = json.loads(out_b[0])
        self.assertEqual(payload_b["last_model_output"], "raw::prompt_b")

        _run_generate_until_agentic(lm, [req_b], response_cache=self.cache)
        self.assertEqual(lm.calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

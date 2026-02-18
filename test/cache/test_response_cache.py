"""Verification tests for the response-level cache system.

Run with: uv run python tests/test_response_cache.py

Tests:
  1. Unit: cache key correctness (collision, determinism, fingerprint)
  2. Integration: mock model -> cache miss on run 1, cache hit on run 2
  3. Crash recovery: JSONL replay after simulated crash
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from lmms_eval.api.instance import Instance
from lmms_eval.caching.response_cache import (
    ResponseCache,
    _extract_content_hash,
    canonicalize_gen_kwargs,
    compute_cache_key,
    extract_gen_kwargs,
    is_deterministic,
)


def _make_instance(request_type, arguments, idx, task_name, doc_id, repeats=1):
    return Instance(
        request_type=request_type,
        arguments=arguments,
        idx=idx,
        metadata={"task": task_name, "doc_id": doc_id, "repeats": repeats},
    )


class TestDeterminismDetection(unittest.TestCase):
    def test_loglikelihood_always_deterministic(self):
        assert is_deterministic("loglikelihood", {"temperature": 999}) is True

    def test_temp_zero_deterministic(self):
        assert is_deterministic("generate_until", {"temperature": 0}) is True
        assert is_deterministic("generate_until", {}) is True
        assert is_deterministic("generate_until", None) is True

    def test_temp_positive_nondeterministic(self):
        assert is_deterministic("generate_until", {"temperature": 0.7}) is False
        assert is_deterministic("generate_until", {"temperature": 1}) is False
        assert is_deterministic("generate_until", {"temperature": 0.01}) is False

    def test_do_sample_nondeterministic(self):
        assert is_deterministic("generate_until", {"temperature": 0, "do_sample": True}) is False

    def test_multi_return_nondeterministic(self):
        assert is_deterministic("generate_until", {"n": 3}) is False
        assert is_deterministic("generate_until", {"best_of": 2}) is False
        assert is_deterministic("generate_until", {"num_return_sequences": 5}) is False


class TestCacheKeyCollision(unittest.TestCase):
    def test_conditional_vs_unconditional_differ(self):
        cond = _make_instance("loglikelihood", ("What is 2+2?", " A) 4", None, 0, "t", "test"), 0, "t", 42)
        uncond = _make_instance("loglikelihood", ("", "A) 4"), 0, "t", 42)
        ch_c = _extract_content_hash(cond)
        ch_u = _extract_content_hash(uncond)
        self.assertNotEqual(ch_c, ch_u)
        k_c = compute_cache_key("loglikelihood", "t", 42, {}, idx=0, content_hash=ch_c)
        k_u = compute_cache_key("loglikelihood", "t", 42, {}, idx=0, content_hash=ch_u)
        self.assertNotEqual(k_c, k_u)

    def test_different_idx_differ(self):
        k0 = compute_cache_key("loglikelihood", "t", 42, {}, idx=0)
        k1 = compute_cache_key("loglikelihood", "t", 42, {}, idx=1)
        self.assertNotEqual(k0, k1)

    def test_float_int_normalize(self):
        k_f = compute_cache_key("generate_until", "t", 1, {"temperature": 0.0})
        k_i = compute_cache_key("generate_until", "t", 1, {"temperature": 0})
        self.assertEqual(k_f, k_i)

    def test_task_fingerprint_invalidates(self):
        k1 = compute_cache_key("generate_until", "t", 1, {}, task_fingerprint="abc")
        k2 = compute_cache_key("generate_until", "t", 1, {}, task_fingerprint="def")
        self.assertNotEqual(k1, k2)


class TestExtractGenKwargs(unittest.TestCase):
    def test_simple_model_layout(self):
        inst = _make_instance("generate_until", ("ctx", {"temperature": 0, "until": ["\n"]}, None, 0, "t", "test"), 0, "t", 1)
        gk = extract_gen_kwargs(inst)
        self.assertEqual(gk["temperature"], 0)

    def test_no_dict_returns_empty(self):
        inst = _make_instance("loglikelihood", ("ctx", "continuation"), 0, "t", 1)
        gk = extract_gen_kwargs(inst)
        self.assertEqual(gk, {})


class TestCacheHitVerification(unittest.TestCase):
    """End-to-end: mock model -> run 1 misses -> run 2 hits."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "rank0.db")
        self.audit_path = os.path.join(self.tmpdir, "rank0.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_requests(self):
        return [
            _make_instance("generate_until", ("prompt1", {"temperature": 0, "until": ["\n"]}, None, 0, "mme", "test"), 0, "mme", 0),
            _make_instance("generate_until", ("prompt2", {"temperature": 0, "until": ["\n"]}, None, 1, "mme", "test"), 0, "mme", 1),
            _make_instance("generate_until", ("prompt3", {"temperature": 0, "until": ["\n"]}, None, 2, "mme", "test"), 0, "mme", 2),
        ]

    def test_run1_miss_run2_hit(self):
        mock_model = MagicMock()
        mock_model.generate_until = MagicMock(return_value=["answer_0", "answer_1", "answer_2"])

        cache = ResponseCache(self.db_path, self.audit_path, model_fingerprint="test_model")

        # Run 1: all misses
        requests = self._make_requests()
        results = cache.execute(mock_model, "generate_until", requests)
        self.assertEqual(results, ["answer_0", "answer_1", "answer_2"])
        mock_model.generate_until.assert_called_once()
        stats1 = cache.get_stats()
        self.assertEqual(stats1["hits"], 0)
        self.assertEqual(stats1["misses"], 3)
        cache.close()

        # Run 2: reopen cache, all hits
        mock_model2 = MagicMock()
        mock_model2.generate_until = MagicMock(return_value=[])

        cache2 = ResponseCache(self.db_path, self.audit_path, model_fingerprint="test_model")
        requests2 = self._make_requests()
        results2 = cache2.execute(mock_model2, "generate_until", requests2)
        self.assertEqual(results2, ["answer_0", "answer_1", "answer_2"])
        mock_model2.generate_until.assert_not_called()
        stats2 = cache2.get_stats()
        self.assertEqual(stats2["hits"], 3)
        self.assertEqual(stats2["misses"], 0)
        cache2.close()

    def test_nondeterministic_skipped(self):
        requests = [
            _make_instance("generate_until", ("p", {"temperature": 0.7}, None, 0, "t", "test"), 0, "t", 0),
        ]
        mock_model = MagicMock()
        mock_model.generate_until = MagicMock(return_value=["stochastic_answer"])

        cache = ResponseCache(self.db_path, self.audit_path)
        results = cache.execute(mock_model, "generate_until", requests)
        self.assertEqual(results, ["stochastic_answer"])
        self.assertEqual(cache.get_stats()["skipped_non_deterministic"], 1)
        self.assertEqual(cache.get_stats()["total_cached_entries"], 0)
        cache.close()

    def test_partial_cache_hit(self):
        mock_model = MagicMock()
        mock_model.generate_until = MagicMock(return_value=["a0", "a1", "a2"])

        cache = ResponseCache(self.db_path, self.audit_path)

        # Populate 3 entries
        requests = self._make_requests()
        cache.execute(mock_model, "generate_until", requests)
        cache.close()

        # Reopen, request 4 items (3 cached + 1 new)
        cache2 = ResponseCache(self.db_path, self.audit_path)
        requests2 = self._make_requests()
        new_req = _make_instance("generate_until", ("prompt_new", {"temperature": 0, "until": ["\n"]}, None, 99, "mme", "test"), 0, "mme", 99)
        requests2.append(new_req)

        mock_model2 = MagicMock()
        mock_model2.generate_until = MagicMock(return_value=["a_new"])
        results = cache2.execute(mock_model2, "generate_until", requests2)
        self.assertEqual(results, ["a0", "a1", "a2", "a_new"])
        mock_model2.generate_until.assert_called_once()
        # Only the new request should be sent to the model
        sent = mock_model2.generate_until.call_args[0][0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].doc_id, 99)
        stats = cache2.get_stats()
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["misses"], 1)
        cache2.close()


class TestCrashRecovery(unittest.TestCase):
    def test_jsonl_replay_after_simulated_crash(self):
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "rank0.db")
            audit_path = os.path.join(tmpdir, "rank0.jsonl")

            mock_model = MagicMock()
            mock_model.generate_until = MagicMock(return_value=["crash_answer"])

            cache = ResponseCache(db_path, audit_path)
            requests = [
                _make_instance("generate_until", ("p", {"temperature": 0}, None, 0, "t", "test"), 0, "t", 0),
            ]
            cache.execute(mock_model, "generate_until", requests)

            # Simulate crash: close audit file but delete SQLite DB
            cache._audit_file.close()
            os.remove(db_path)
            wal = db_path + "-wal"
            shm = db_path + "-shm"
            if os.path.exists(wal):
                os.remove(wal)
            if os.path.exists(shm):
                os.remove(shm)

            # Reopen: should replay from JSONL
            cache2 = ResponseCache(db_path, audit_path)
            key = compute_cache_key("generate_until", "t", 0, {"temperature": 0}, idx=0)
            result = cache2._lookup(key)
            self.assertEqual(result, "crash_answer")
            cache2.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPoisoningPrevention(unittest.TestCase):
    def test_none_rejected(self):
        self.assertFalse(ResponseCache._is_valid_response(None, "generate_until"))

    def test_empty_string_rejected(self):
        self.assertFalse(ResponseCache._is_valid_response("", "generate_until"))
        self.assertFalse(ResponseCache._is_valid_response("  ", "generate_until"))

    def test_malformed_loglikelihood_rejected(self):
        self.assertFalse(ResponseCache._is_valid_response([0.5], "loglikelihood"))
        self.assertFalse(ResponseCache._is_valid_response("string", "loglikelihood"))

    def test_valid_responses_accepted(self):
        self.assertTrue(ResponseCache._is_valid_response("answer", "generate_until"))
        self.assertTrue(ResponseCache._is_valid_response([0.5, True], "loglikelihood"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

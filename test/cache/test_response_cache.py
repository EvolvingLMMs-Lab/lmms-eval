import json
import os
import shutil
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from lmms_eval.api.instance import Instance
from lmms_eval.caching.response_cache import (
    ResponseCache,
    _extract_content_hash,
    compute_cache_key,
    extract_gen_kwargs,
    is_deterministic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(request_type, arguments, idx, task_name, doc_id, repeats=1):
    return Instance(request_type=request_type, arguments=arguments, idx=idx, metadata={"task": task_name, "doc_id": doc_id, "repeats": repeats})


def _gen_request(prompt="prompt", doc_id=0, idx=0, task="t", temperature=0, do_sample=None, n=None, until=None, repeats=1):
    gk = {"temperature": temperature, "until": until or ["\n"]}
    if do_sample is not None:
        gk["do_sample"] = do_sample
    if n is not None:
        gk["n"] = n
    return _make_instance("generate_until", (prompt, gk, None, doc_id, task, "test"), idx, task, doc_id, repeats)


def _mock_model(responses):
    m = MagicMock()
    m.generate_until = MagicMock(return_value=responses)
    return m


class _CacheTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "rank0.db")
        self.audit_path = os.path.join(self.tmpdir, "rank0.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_cache(self, **kwargs):
        return ResponseCache(self.db_path, self.audit_path, **kwargs)

    def _read_audit_lines(self):
        with open(self.audit_path) as f:
            return [json.loads(line) for line in f if line.strip()]


# ===========================================================================
# Unit tests - pure functions, no I/O
# ===========================================================================


class TestDeterminismDetection(unittest.TestCase):
    def test_loglikelihood_always_deterministic(self):
        self.assertTrue(is_deterministic("loglikelihood", {"temperature": 999}))

    def test_temp_zero_deterministic(self):
        self.assertTrue(is_deterministic("generate_until", {"temperature": 0}))
        self.assertTrue(is_deterministic("generate_until", {}))
        self.assertTrue(is_deterministic("generate_until", None))

    def test_temp_positive_nondeterministic(self):
        self.assertFalse(is_deterministic("generate_until", {"temperature": 0.7}))
        self.assertFalse(is_deterministic("generate_until", {"temperature": 1}))
        self.assertFalse(is_deterministic("generate_until", {"temperature": 0.01}))

    def test_do_sample_nondeterministic(self):
        self.assertFalse(is_deterministic("generate_until", {"temperature": 0, "do_sample": True}))

    def test_multi_return_nondeterministic(self):
        self.assertFalse(is_deterministic("generate_until", {"n": 3}))
        self.assertFalse(is_deterministic("generate_until", {"best_of": 2}))
        self.assertFalse(is_deterministic("generate_until", {"num_return_sequences": 5}))


class TestCacheKeyCollision(unittest.TestCase):
    def test_conditional_vs_unconditional_differ(self):
        cond = _make_instance("loglikelihood", ("What is 2+2?", " A) 4", None, 0, "t", "test"), 0, "t", 42)
        uncond = _make_instance("loglikelihood", ("", "A) 4"), 0, "t", 42)
        ch_c, ch_u = _extract_content_hash(cond), _extract_content_hash(uncond)
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
        gk = extract_gen_kwargs(_gen_request(temperature=0))
        self.assertEqual(gk["temperature"], 0)

    def test_no_dict_returns_empty(self):
        inst = _make_instance("loglikelihood", ("ctx", "continuation"), 0, "t", 1)
        self.assertEqual(extract_gen_kwargs(inst), {})


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


# ===========================================================================
# Integration - cache hit/miss
# ===========================================================================


class TestCacheHitMiss(_CacheTestBase):
    def _make_3_det_requests(self, task="mme"):
        return [_gen_request(f"prompt{i}", doc_id=i, task=task) for i in range(3)]

    def test_run1_miss_run2_hit(self):
        model = _mock_model(["a0", "a1", "a2"])
        cache = self._open_cache(model_fingerprint="test_model")
        results = cache.execute(model, "generate_until", self._make_3_det_requests())
        self.assertEqual(results, ["a0", "a1", "a2"])
        model.generate_until.assert_called_once()
        self.assertEqual(cache.get_stats()["hits"], 0)
        self.assertEqual(cache.get_stats()["misses"], 3)
        cache.close()

        model2 = _mock_model([])
        cache2 = self._open_cache(model_fingerprint="test_model")
        results2 = cache2.execute(model2, "generate_until", self._make_3_det_requests())
        self.assertEqual(results2, ["a0", "a1", "a2"])
        model2.generate_until.assert_not_called()
        self.assertEqual(cache2.get_stats()["hits"], 3)
        cache2.close()

    def test_partial_cache_hit(self):
        model = _mock_model(["a0", "a1", "a2"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", self._make_3_det_requests())
        cache.close()

        reqs = self._make_3_det_requests()
        reqs.append(_gen_request("prompt_new", doc_id=99, task="mme"))
        model2 = _mock_model(["a_new"])
        cache2 = self._open_cache()
        results = cache2.execute(model2, "generate_until", reqs)
        self.assertEqual(results, ["a0", "a1", "a2", "a_new"])
        sent = model2.generate_until.call_args[0][0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].doc_id, 99)
        self.assertEqual(cache2.get_stats()["hits"], 3)
        self.assertEqual(cache2.get_stats()["misses"], 1)
        cache2.close()


# ===========================================================================
# Integration - non-deterministic requests must bypass cache
# ===========================================================================


class TestNonDeterministicBypass(_CacheTestBase):
    def test_skipped_on_positive_temperature(self):
        model = _mock_model(["stochastic_answer"])
        cache = self._open_cache()
        results = cache.execute(model, "generate_until", [_gen_request(temperature=0.7)])
        self.assertEqual(results, ["stochastic_answer"])
        self.assertEqual(cache.get_stats()["skipped_non_deterministic"], 1)
        self.assertEqual(cache.get_stats()["total_cached_entries"], 0)
        cache.close()

    def test_temp_positive_repeats_all_bypass(self):
        reqs = [_gen_request(temperature=0.7, repeats=3)] * 3
        model = _mock_model(["ans_0", "ans_1", "ans_2"])
        cache = self._open_cache()
        results = cache.execute(model, "generate_until", reqs)
        self.assertEqual(results, ["ans_0", "ans_1", "ans_2"])
        self.assertEqual(len(model.generate_until.call_args[0][0]), 3)
        self.assertEqual(cache.get_stats()["skipped_non_deterministic"], 3)
        self.assertEqual(cache.get_stats()["total_cached_entries"], 0)
        cache.close()

    def test_temp_zero_repeats_all_hit_same_entry(self):
        base = _gen_request(temperature=0, repeats=3)
        model = _mock_model(["deterministic_answer"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", [base])
        cache.close()

        model2 = _mock_model([])
        cache2 = self._open_cache()
        results = cache2.execute(model2, "generate_until", [base] * 3)
        self.assertEqual(results, ["deterministic_answer"] * 3)
        model2.generate_until.assert_not_called()
        self.assertEqual(cache2.get_stats()["hits"], 3)
        cache2.close()

    def test_do_sample_repeats_bypass(self):
        reqs = [_gen_request(temperature=0, do_sample=True, repeats=3)] * 3
        model = _mock_model(["s0", "s1", "s2"])
        cache = self._open_cache()
        results = cache.execute(model, "generate_until", reqs)
        self.assertEqual(results, ["s0", "s1", "s2"])
        self.assertEqual(len(model.generate_until.call_args[0][0]), 3)
        self.assertEqual(cache.get_stats()["skipped_non_deterministic"], 3)
        self.assertEqual(cache.get_stats()["total_cached_entries"], 0)
        cache.close()

    def test_no_cross_run_poisoning(self):
        model = _mock_model(["run1_answer"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", [_gen_request(temperature=0.7)])
        self.assertEqual(cache.get_stats()["total_cached_entries"], 0)
        cache.close()

        model2 = _mock_model(["run2_answer"])
        cache2 = self._open_cache()
        results = cache2.execute(model2, "generate_until", [_gen_request(temperature=0.7)])
        self.assertEqual(results, ["run2_answer"])
        model2.generate_until.assert_called_once()
        self.assertEqual(cache2.get_stats()["total_cached_entries"], 0)
        cache2.close()


# ===========================================================================
# Integration - JSONL audit log observability
# ===========================================================================


class TestAuditLogObservability(_CacheTestBase):
    def test_nondeterministic_responses_logged(self):
        reqs = [_gen_request(f"p{i}", doc_id=i, temperature=0.9) for i in range(2)]
        model = _mock_model(["stoch_0", "stoch_1"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", reqs)
        self.assertEqual(cache.get_stats()["total_cached_entries"], 0)
        cache.close()

        lines = self._read_audit_lines()
        self.assertEqual(len(lines), 2)
        self.assertFalse(lines[0]["deterministic"])
        self.assertFalse(lines[1]["deterministic"])
        self.assertEqual(json.loads(lines[0]["response"]), "stoch_0")
        self.assertEqual(json.loads(lines[1]["response"]), "stoch_1")

    def test_deterministic_responses_logged(self):
        model = _mock_model(["det_answer"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", [_gen_request(temperature=0)])
        self.assertEqual(cache.get_stats()["total_cached_entries"], 1)
        cache.close()

        lines = self._read_audit_lines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0]["deterministic"])
        self.assertEqual(json.loads(lines[0]["response"]), "det_answer")
        self.assertNotEqual(lines[0]["cache_key"], "")

    def test_mixed_all_logged(self):
        reqs = [
            _gen_request("p1", doc_id=0, temperature=0),
            _gen_request("p2", doc_id=1, temperature=0.8),
        ]
        model = _mock_model(["det", "stoch"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", reqs)
        cache.close()

        lines = self._read_audit_lines()
        self.assertEqual(len(lines), 2)
        det_lines = [l for l in lines if l["deterministic"]]
        nondet_lines = [l for l in lines if not l["deterministic"]]
        self.assertEqual(len(det_lines), 1)
        self.assertEqual(len(nondet_lines), 1)


# ===========================================================================
# Integration - crash recovery via JSONL replay
# ===========================================================================


class TestCrashRecovery(_CacheTestBase):
    def test_jsonl_replay_after_simulated_crash(self):
        model = _mock_model(["crash_answer"])
        cache = self._open_cache()
        cache.execute(model, "generate_until", [_gen_request(temperature=0)])

        cache._audit_file.close()
        os.remove(self.db_path)
        for suffix in ("-wal", "-shm"):
            p = self.db_path + suffix
            if os.path.exists(p):
                os.remove(p)

        cache2 = self._open_cache()
        key = compute_cache_key("generate_until", "t", 0, {"temperature": 0, "until": ["\n"]}, idx=0)
        self.assertEqual(cache2._lookup(key), "crash_answer")
        cache2.close()


# ===========================================================================
# Integration - multi-rank isolation and shard merging
# ===========================================================================


class TestMultiRankIsolation(_CacheTestBase):
    def test_separate_ranks_write_independently(self):
        db0 = os.path.join(self.tmpdir, "rank0.db")
        audit0 = os.path.join(self.tmpdir, "rank0.jsonl")
        db1 = os.path.join(self.tmpdir, "rank1.db")
        audit1 = os.path.join(self.tmpdir, "rank1.jsonl")

        cache0 = ResponseCache(db0, audit0, model_fingerprint="model_A")
        cache0.execute(_mock_model(["r0_a0", "r0_a1"]), "generate_until", [_gen_request(f"p{i}", doc_id=i) for i in range(2)])
        self.assertEqual(cache0.get_stats()["total_cached_entries"], 2)
        cache0.close()

        cache1 = ResponseCache(db1, audit1, model_fingerprint="model_A")
        cache1.execute(_mock_model(["r1_a2", "r1_a3"]), "generate_until", [_gen_request(f"p{i}", doc_id=i) for i in range(2, 4)])
        self.assertEqual(cache1.get_stats()["total_cached_entries"], 2)
        cache1.close()

        cache0_reopen = ResponseCache(db0, audit0)
        self.assertEqual(cache0_reopen.get_stats()["total_cached_entries"], 2)
        cache0_reopen.close()

    def test_merge_shards_combines_ranks(self):
        db0 = os.path.join(self.tmpdir, "rank0.db")
        audit0 = os.path.join(self.tmpdir, "rank0.jsonl")
        db1 = os.path.join(self.tmpdir, "rank1.db")
        audit1 = os.path.join(self.tmpdir, "rank1.jsonl")

        cache0 = ResponseCache(db0, audit0)
        cache0.execute(_mock_model(["a0"]), "generate_until", [_gen_request("p0", doc_id=0)])
        cache0.close()

        cache1 = ResponseCache(db1, audit1)
        cache1.execute(_mock_model(["a1"]), "generate_until", [_gen_request("p1", doc_id=1)])
        cache1.close()

        merged_path = os.path.join(self.tmpdir, "merged.db")
        ResponseCache.merge_shards([db0, db1], merged_path)

        merged_db = sqlite3.connect(merged_path)
        count = merged_db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        self.assertEqual(count, 2)
        merged_db.close()

    def test_merge_shards_deduplicates_overlapping_keys(self):
        db0 = os.path.join(self.tmpdir, "rank0.db")
        audit0 = os.path.join(self.tmpdir, "rank0.jsonl")
        db1 = os.path.join(self.tmpdir, "rank1.db")
        audit1 = os.path.join(self.tmpdir, "rank1.jsonl")

        req = _gen_request("same_prompt", doc_id=0)
        cache0 = ResponseCache(db0, audit0)
        cache0.execute(_mock_model(["answer"]), "generate_until", [req])
        cache0.close()

        cache1 = ResponseCache(db1, audit1)
        cache1.execute(_mock_model(["answer"]), "generate_until", [req])
        cache1.close()

        merged_path = os.path.join(self.tmpdir, "merged.db")
        ResponseCache.merge_shards([db0, db1], merged_path)

        merged_db = sqlite3.connect(merged_path)
        count = merged_db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        self.assertEqual(count, 1)
        merged_db.close()


# ===========================================================================
# Integration - model fingerprint isolation
# ===========================================================================


class TestModelFingerprintIsolation(_CacheTestBase):
    def test_fingerprint_stored_in_meta(self):
        cache = self._open_cache(model_fingerprint="model_A|args_A")
        row = cache.db.execute("SELECT value FROM meta WHERE key = 'model_fingerprint'").fetchone()
        self.assertEqual(row[0], "model_A|args_A")
        cache.close()

    def test_different_models_use_separate_db_files(self):
        db_a = os.path.join(self.tmpdir, "model_a.db")
        audit_a = os.path.join(self.tmpdir, "model_a.jsonl")
        db_b = os.path.join(self.tmpdir, "model_b.db")
        audit_b = os.path.join(self.tmpdir, "model_b.jsonl")

        cache_a = ResponseCache(db_a, audit_a, model_fingerprint="model_A")
        cache_a.execute(_mock_model(["answer_A"]), "generate_until", [_gen_request("prompt", doc_id=0)])
        self.assertEqual(cache_a.get_stats()["total_cached_entries"], 1)
        cache_a.close()

        model_b = _mock_model(["answer_B"])
        cache_b = ResponseCache(db_b, audit_b, model_fingerprint="model_B")
        results = cache_b.execute(model_b, "generate_until", [_gen_request("prompt", doc_id=0)])
        self.assertEqual(results, ["answer_B"])
        model_b.generate_until.assert_called_once()
        self.assertEqual(cache_b.get_stats()["hits"], 0)
        self.assertEqual(cache_b.get_stats()["misses"], 1)
        cache_b.close()


# ===========================================================================
# Integration - stats accuracy across lifecycle
# ===========================================================================


class TestStatsAccuracy(_CacheTestBase):
    def test_total_cached_entries_across_close_reopen(self):
        cache = self._open_cache()
        reqs = [_gen_request(f"p{i}", doc_id=i) for i in range(5)]
        cache.execute(_mock_model([f"a{i}" for i in range(5)]), "generate_until", reqs)
        self.assertEqual(cache.get_stats()["total_cached_entries"], 5)
        cache.close()

        cache2 = self._open_cache()
        self.assertEqual(cache2.get_stats()["total_cached_entries"], 5)
        new_reqs = [_gen_request(f"p{i}", doc_id=i) for i in range(5, 7)]
        cache2.execute(_mock_model(["a5", "a6"]), "generate_until", new_reqs)
        self.assertEqual(cache2.get_stats()["total_cached_entries"], 7)
        cache2.close()

        cache3 = self._open_cache()
        self.assertEqual(cache3.get_stats()["total_cached_entries"], 7)
        cache3.close()

    def test_hit_miss_counters_reset_on_new_instance(self):
        cache = self._open_cache()
        cache.execute(_mock_model(["a0"]), "generate_until", [_gen_request(temperature=0)])
        self.assertEqual(cache.get_stats()["misses"], 1)
        cache.close()

        cache2 = self._open_cache()
        self.assertEqual(cache2.get_stats()["hits"], 0)
        self.assertEqual(cache2.get_stats()["misses"], 0)
        self.assertEqual(cache2.get_stats()["skipped_non_deterministic"], 0)
        cache2.close()


# ===========================================================================
# Integration - large batch sanity
# NOTE: loglikelihood execute flow is NOT covered by these tests.
# ===========================================================================


class TestLargeBatchSanity(_CacheTestBase):
    def test_1000_requests_write_and_read(self):
        n = 1000
        reqs = [_gen_request(f"prompt_{i}", doc_id=i) for i in range(n)]
        responses = [f"answer_{i}" for i in range(n)]

        model = _mock_model(responses)
        cache = self._open_cache()
        t0 = time.monotonic()
        results = cache.execute(model, "generate_until", reqs)
        elapsed_write = time.monotonic() - t0
        self.assertEqual(len(results), n)
        self.assertEqual(cache.get_stats()["total_cached_entries"], n)
        cache.close()

        model2 = _mock_model([])
        cache2 = self._open_cache()
        t0 = time.monotonic()
        results2 = cache2.execute(model2, "generate_until", reqs)
        elapsed_read = time.monotonic() - t0
        self.assertEqual(results2, responses)
        model2.generate_until.assert_not_called()
        self.assertEqual(cache2.get_stats()["hits"], n)
        self.assertLess(elapsed_write, 10.0)
        self.assertLess(elapsed_read, 10.0)
        cache2.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

import os
import unittest

import pytest

from lmms_eval.evaluator import simple_evaluate

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_VERSION = "bytedance-seed/seed-1.6-flash"
SKIP_REASON = "OPENROUTER_API_KEY not set"


def _run_eval(limit=2, max_tokens=None):
    old_key = os.environ.get("OPENAI_API_KEY")
    old_base = os.environ.get("OPENAI_API_BASE")
    try:
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        return simple_evaluate(
            model="openai",
            model_args=f"model_version={MODEL_VERSION}",
            tasks=["mme"],
            batch_size=1,
            limit=limit,
            max_tokens=max_tokens,
        )
    finally:
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
        if old_base is not None:
            os.environ["OPENAI_API_BASE"] = old_base
        else:
            os.environ.pop("OPENAI_API_BASE", None)


@pytest.mark.api
@pytest.mark.slow
@unittest.skipUnless(OPENROUTER_API_KEY, SKIP_REASON)
class TestUsageTrackingE2E(unittest.TestCase):

    _cached_results = None

    @classmethod
    def _get_results(cls):
        if cls._cached_results is None:
            cls._cached_results = _run_eval(limit=2)
        return cls._cached_results

    def test_usage_key_always_present(self):
        results = self._get_results()
        self.assertIn("usage", results)

    def test_usage_has_expected_structure(self):
        usage = self._get_results()["usage"]
        self.assertIn("total", usage)
        self.assertIn("by_task", usage)
        self.assertIn("by_source", usage)
        self.assertIn("budget_exceeded", usage)
        self.assertIn("budget_total_tokens", usage)

    def test_token_counts_are_positive(self):
        total = self._get_results()["usage"]["total"]
        self.assertGreater(total["input_tokens"], 0)
        self.assertGreater(total["output_tokens"], 0)
        self.assertGreater(total["total_tokens"], 0)
        self.assertGreater(total["n_api_calls"], 0)

    def test_api_calls_match_limit(self):
        total = self._get_results()["usage"]["total"]
        self.assertEqual(total["n_api_calls"], 2)

    def test_total_equals_sum_of_parts(self):
        total = self._get_results()["usage"]["total"]
        self.assertEqual(
            total["total_tokens"],
            total["input_tokens"] + total["output_tokens"] + total["reasoning_tokens"],
        )

    def test_by_source_has_model(self):
        by_source = self._get_results()["usage"]["by_source"]
        self.assertIn("model", by_source)
        self.assertGreater(by_source["model"]["n_api_calls"], 0)

    def test_no_budget_means_not_exceeded(self):
        usage = self._get_results()["usage"]
        self.assertFalse(usage["budget_exceeded"])
        self.assertIsNone(usage["budget_total_tokens"])


@pytest.mark.api
@pytest.mark.slow
@unittest.skipUnless(OPENROUTER_API_KEY, SKIP_REASON)
class TestBudgetEnforcement(unittest.TestCase):

    def test_budget_exceeded_with_tiny_limit(self):
        results = _run_eval(limit=2, max_tokens=1)
        usage = results["usage"]
        self.assertTrue(usage["budget_exceeded"])
        self.assertEqual(usage["budget_total_tokens"], 1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for lmms_eval.models.model_utils.usage_metrics."""

import threading
import unittest

from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
    reset_usage_metrics,
    set_budget,
    set_task_context,
    summarize_usage_metrics,
)


class TestUsageMetrics(unittest.TestCase):

    def setUp(self):
        reset_usage_metrics()

    def tearDown(self):
        reset_usage_metrics()

    def test_log_and_summarize_single_call(self):
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50, reasoning_tokens=10, source="model")

        summary = summarize_usage_metrics()
        self.assertEqual(summary["total"]["input_tokens"], 100)
        self.assertEqual(summary["total"]["output_tokens"], 50)
        self.assertEqual(summary["total"]["reasoning_tokens"], 10)
        self.assertEqual(summary["total"]["total_tokens"], 160)
        self.assertEqual(summary["total"]["n_api_calls"], 1)

    def test_log_and_summarize_multiple_calls(self):
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50)
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=200, output_tokens=80, reasoning_tokens=20)

        summary = summarize_usage_metrics()
        self.assertEqual(summary["total"]["input_tokens"], 300)
        self.assertEqual(summary["total"]["output_tokens"], 130)
        self.assertEqual(summary["total"]["reasoning_tokens"], 20)
        self.assertEqual(summary["total"]["total_tokens"], 450)
        self.assertEqual(summary["total"]["n_api_calls"], 2)

    def test_summarize_empty_returns_empty_dict(self):
        summary = summarize_usage_metrics()
        self.assertEqual(summary, {})

    def test_running_totals_empty(self):
        totals = get_running_totals()
        self.assertEqual(totals["total_tokens"], 0)
        self.assertEqual(totals["n_api_calls"], 0)

    def test_by_task_aggregation(self):
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50)
        log_usage(model_name="gpt-4o", task_name="mmmu_val", input_tokens=200, output_tokens=80)
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=150, output_tokens=60)

        summary = summarize_usage_metrics()
        self.assertIn("mme", summary["by_task"])
        self.assertIn("mmmu_val", summary["by_task"])

        mme = summary["by_task"]["mme"]
        self.assertEqual(mme["input_tokens"], 250)
        self.assertEqual(mme["output_tokens"], 110)
        self.assertEqual(mme["n_api_calls"], 2)

        mmmu = summary["by_task"]["mmmu_val"]
        self.assertEqual(mmmu["input_tokens"], 200)
        self.assertEqual(mmmu["output_tokens"], 80)
        self.assertEqual(mmmu["n_api_calls"], 1)

    def test_by_task_unknown_fallback(self):
        log_usage(model_name="gpt-4o", task_name=None, input_tokens=50, output_tokens=25)

        summary = summarize_usage_metrics()
        self.assertIn("_unknown", summary["by_task"])
        self.assertEqual(summary["by_task"]["_unknown"]["n_api_calls"], 1)

    def test_by_source_aggregation(self):
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50, source="model")
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=80, output_tokens=40, source="judge")
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=120, output_tokens=60, source="model")

        summary = summarize_usage_metrics()
        self.assertIn("model", summary["by_source"])
        self.assertIn("judge", summary["by_source"])

        model_agg = summary["by_source"]["model"]
        self.assertEqual(model_agg["input_tokens"], 220)
        self.assertEqual(model_agg["n_api_calls"], 2)

        judge_agg = summary["by_source"]["judge"]
        self.assertEqual(judge_agg["input_tokens"], 80)
        self.assertEqual(judge_agg["n_api_calls"], 1)

    def test_budget_not_set_never_exceeded(self):
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=999999, output_tokens=999999)
        self.assertFalse(is_budget_exceeded())

    def test_budget_exceeded(self):
        set_budget(max_tokens=500)
        self.assertFalse(is_budget_exceeded())

        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=300, output_tokens=100)
        self.assertFalse(is_budget_exceeded())

        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=50, output_tokens=60)
        self.assertTrue(is_budget_exceeded())

    def test_budget_exact_boundary(self):
        set_budget(max_tokens=200)

        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=100)
        self.assertTrue(is_budget_exceeded())

    def test_budget_includes_reasoning_tokens(self):
        set_budget(max_tokens=100)

        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=30, output_tokens=30, reasoning_tokens=50)
        self.assertTrue(is_budget_exceeded())

    def test_budget_exceeded_in_summary(self):
        set_budget(max_tokens=100)
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=200, output_tokens=50)

        summary = summarize_usage_metrics()
        self.assertTrue(summary["budget_exceeded"])
        self.assertEqual(summary["budget_total_tokens"], 100)

    def test_budget_not_exceeded_in_summary(self):
        set_budget(max_tokens=1000)
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50)

        summary = summarize_usage_metrics()
        self.assertFalse(summary["budget_exceeded"])
        self.assertEqual(summary["budget_total_tokens"], 1000)

    def test_reset_clears_everything(self):
        set_budget(max_tokens=100)
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=200, output_tokens=50)
        self.assertTrue(is_budget_exceeded())

        reset_usage_metrics()

        self.assertFalse(is_budget_exceeded())
        self.assertEqual(summarize_usage_metrics(), {})
        self.assertEqual(get_running_totals()["total_tokens"], 0)

    def test_task_context_fallback(self):
        set_task_context("mmmu_val")
        log_usage(model_name="gpt-4o", task_name=None, input_tokens=100, output_tokens=50, source="judge")

        summary = summarize_usage_metrics()
        self.assertIn("mmmu_val", summary["by_task"])
        self.assertNotIn("_unknown", summary["by_task"])

    def test_task_context_override(self):
        set_task_context("mmmu_val")
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50)

        summary = summarize_usage_metrics()
        self.assertIn("mme", summary["by_task"])
        self.assertNotIn("mmmu_val", summary["by_task"])

    def test_task_context_cleared(self):
        set_task_context("mmmu_val")
        set_task_context(None)
        log_usage(model_name="gpt-4o", task_name=None, input_tokens=100, output_tokens=50)

        summary = summarize_usage_metrics()
        self.assertIn("_unknown", summary["by_task"])

    def test_running_totals(self):
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=100, output_tokens=50, reasoning_tokens=10)
        log_usage(model_name="gpt-4o", task_name="mme", input_tokens=200, output_tokens=80, reasoning_tokens=20)

        totals = get_running_totals()
        self.assertEqual(totals["input_tokens"], 300)
        self.assertEqual(totals["output_tokens"], 130)
        self.assertEqual(totals["reasoning_tokens"], 30)
        self.assertEqual(totals["total_tokens"], 460)
        self.assertEqual(totals["n_api_calls"], 2)

    def test_concurrent_logging(self):
        n_threads = 8
        calls_per_thread = 100

        def worker():
            for _ in range(calls_per_thread):
                log_usage(model_name="gpt-4o", task_name="mme", input_tokens=1, output_tokens=1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = summarize_usage_metrics()
        expected_calls = n_threads * calls_per_thread
        self.assertEqual(summary["total"]["n_api_calls"], expected_calls)
        self.assertEqual(summary["total"]["input_tokens"], expected_calls)
        self.assertEqual(summary["total"]["output_tokens"], expected_calls)


if __name__ == "__main__":
    unittest.main()

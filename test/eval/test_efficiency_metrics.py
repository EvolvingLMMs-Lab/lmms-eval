import unittest

from lmms_eval.models.model_utils.efficiency_metrics import build_efficiency_summary


class TestEfficiencyMetrics(unittest.TestCase):
    def _base_results(self):
        return {
            "config": {
                "model": "openai",
                "model_args": "model_version=gpt-5.2-mini",
            },
            "configs": {
                "toy_task": {
                    "score_key": "score",
                }
            },
            "samples": {
                "toy_task": [
                    {
                        "token_counts": [{"input_tokens": 100, "output_tokens": 20}],
                        "score": 1,
                    },
                    {
                        "token_counts": [{"input_tokens": 50, "output_tokens": 30}],
                        "score": 0,
                    },
                ]
            },
        }

    def test_tokens_per_correct_without_pricing(self):
        summary = build_efficiency_summary(self._base_results())
        self.assertIn("overall", summary)
        self.assertEqual(summary["overall"]["total_input_tokens"], 150.0)
        self.assertEqual(summary["overall"]["total_output_tokens"], 50.0)
        self.assertEqual(summary["overall"]["total_correct_score"], 1.0)
        self.assertEqual(summary["overall"]["tokens_per_correct_answer"], 50.0)
        self.assertNotIn("pricing", summary)

    def test_empty_samples_returns_empty_summary(self):
        summary = build_efficiency_summary({"samples": {}})
        self.assertEqual(summary, {})

    def test_score_fallback_from_acc_when_score_key_missing(self):
        results = self._base_results()
        results["configs"]["toy_task"]["score_key"] = "custom_metric"
        results["samples"]["toy_task"][0].pop("score")
        results["samples"]["toy_task"][0]["acc"] = 1
        results["samples"]["toy_task"][1].pop("score")
        results["samples"]["toy_task"][1]["acc"] = 0

        summary = build_efficiency_summary(results)
        self.assertEqual(summary["overall"]["total_correct_score"], 1.0)
        self.assertEqual(summary["overall"]["tokens_per_correct_answer"], 50.0)

    def test_ignores_non_dict_token_entries(self):
        results = self._base_results()
        results["samples"]["toy_task"][0]["token_counts"] = [None, {"input_tokens": 100, "output_tokens": 20}]
        results["samples"]["toy_task"][1]["token_counts"] = ["bad", {"input_tokens": 50, "output_tokens": 30}]

        summary = build_efficiency_summary(results)
        self.assertEqual(summary["overall"]["total_input_tokens"], 150.0)
        self.assertEqual(summary["overall"]["total_output_tokens"], 50.0)

    def test_zero_correct_score_returns_none_tokens_per_correct(self):
        results = self._base_results()
        results["samples"]["toy_task"][0]["score"] = 0
        results["samples"]["toy_task"][1]["score"] = 0

        summary = build_efficiency_summary(results)
        self.assertEqual(summary["overall"]["total_correct_score"], 0.0)
        self.assertIsNone(summary["overall"]["tokens_per_correct_answer"])


if __name__ == "__main__":
    unittest.main()

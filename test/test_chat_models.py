"""
Integration tests for chat models with throughput metrics
"""
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestChatModelThroughput(unittest.TestCase):
    """Test throughput metrics integration in chat models"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_logger = Mock()

    @patch("lmms_eval.models.chat.openai_compatible.eval_logger")
    @patch("lmms_eval.models.chat.openai_compatible.OpenAI")
    def test_openai_compatible_metrics(self, mock_openai, mock_logger):
        """Test OpenAI compatible model throughput metrics"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens = 5

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Import after mocking
        from lmms_eval.models.chat.openai_compatible import OpenAICompatible

        model = OpenAICompatible(model_version="test-model")
        model.client = mock_client

        # Create mock request
        mock_request = Mock()
        mock_request.args = (
            "test context",
            lambda x: [{"role": "user", "content": "test"}],
            {"max_new_tokens": 100},
            0,
            "test_task",
            "test",
        )

        # Test generate_until
        result = model.generate_until([mock_request])

        # Verify metrics logging was called
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        # Check that throughput metrics were logged
        metrics_logged = any("Inference metrics" in call for call in log_calls)
        self.assertTrue(metrics_logged, "Throughput metrics should be logged")

    def test_timing_integration(self):
        """Test that timing measurements are integrated properly"""

        class MockModel:
            def __init__(self):
                self.generate_call_count = 0

            def generate_with_timing(self):
                """Simulate model generation with timing"""
                self.generate_call_count += 1
                start_time = time.time()
                time.sleep(0.01)  # Simulate processing
                end_time = time.time()

                e2e_latency = end_time - start_time
                output_tokens = 25
                ttft = e2e_latency * 0.1

                if output_tokens > 1:
                    tpot = (e2e_latency - ttft) / (output_tokens - 1)
                    inference_speed = 1 / tpot if tpot > 0 else 0
                else:
                    tpot = e2e_latency
                    inference_speed = 0

                return {
                    "e2e_latency": e2e_latency,
                    "tpot": tpot,
                    "inference_speed": inference_speed,
                    "output_tokens": output_tokens,
                }

        mock_model = MockModel()
        result = mock_model.generate_with_timing()

        # Verify metrics are calculated
        self.assertIn("e2e_latency", result)
        self.assertIn("tpot", result)
        self.assertIn("inference_speed", result)
        self.assertIn("output_tokens", result)

        # Verify reasonable values
        self.assertGreater(result["e2e_latency"], 0)
        self.assertGreater(result["tpot"], 0)
        self.assertGreater(result["inference_speed"], 0)
        self.assertEqual(result["output_tokens"], 25)

    def test_batch_processing_metrics(self):
        """Test batch processing throughput metrics"""

        def calculate_batch_metrics(batch_responses, e2e_latency):
            """Calculate metrics for a batch of responses"""
            total_tokens = sum(len(response.split()) for response in batch_responses)
            batch_size = len(batch_responses)

            if batch_size > 0:
                avg_tokens_per_response = total_tokens / batch_size
                avg_latency_per_response = e2e_latency / batch_size

                ttft_estimate = avg_latency_per_response * 0.1

                if avg_tokens_per_response > 1:
                    tpot = (avg_latency_per_response - ttft_estimate) / (avg_tokens_per_response - 1)
                    inference_speed = 1 / tpot if tpot > 0 else 0
                else:
                    tpot = avg_latency_per_response
                    inference_speed = 0

                return {
                    "total_tokens": total_tokens,
                    "avg_tpot": tpot,
                    "avg_speed": inference_speed,
                    "batch_size": batch_size,
                }
            return {}

        # Test with sample batch
        batch_responses = [
            "This is a test response with several words",
            "Another response that is slightly longer than the first",
            "Short response",
        ]
        e2e_latency = 1.5

        metrics = calculate_batch_metrics(batch_responses, e2e_latency)

        # Verify batch metrics
        self.assertEqual(metrics["batch_size"], 3)
        self.assertGreater(metrics["total_tokens"], 0)
        self.assertGreater(metrics["avg_tpot"], 0)
        self.assertGreater(metrics["avg_speed"], 0)

    @patch("time.time")
    def test_timing_accuracy(self, mock_time):
        """Test timing measurement accuracy with controlled time"""
        # Mock time to return predictable values
        mock_time.side_effect = [0.0, 1.0]  # 1 second elapsed

        start_time = time.time()
        end_time = time.time()
        e2e_latency = end_time - start_time

        self.assertEqual(e2e_latency, 1.0)

        # Test TPOT calculation with known timing
        output_tokens = 20
        ttft = 0.1

        if output_tokens > 1:
            tpot = (e2e_latency - ttft) / (output_tokens - 1)
            inference_speed = 1 / tpot

        expected_tpot = (1.0 - 0.1) / (20 - 1)  # 0.047
        expected_speed = 1 / expected_tpot  # 21.11

        self.assertAlmostEqual(tpot, expected_tpot, places=3)
        self.assertAlmostEqual(inference_speed, expected_speed, places=1)


if __name__ == "__main__":
    unittest.main()

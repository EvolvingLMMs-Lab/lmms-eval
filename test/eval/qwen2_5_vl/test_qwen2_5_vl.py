"""
Test cases for Qwen2.5-VL model evaluation via HTTP server.
"""

import unittest
from unittest import TestCase

from utils import get_gpu_count, run_evaluation_test, with_server, with_temp_dir


class TestQwen2_5_VL(TestCase):
    """Test cases for Qwen2.5-VL model evaluation."""

    @with_temp_dir
    @with_server
    def test_qwen2_5_vl_mme(self, temp_dir, server):
        """Test Qwen2.5-VL evaluation on MME benchmark."""

        num_gpus = get_gpu_count()
        print(f"[TEST] Using {num_gpus} GPUs for evaluation")

        result = run_evaluation_test(
            server_url=server.url,
            model="qwen2_5_vl",
            tasks=["mme"],
            model_args={
                "pretrained": "Qwen/Qwen2.5-VL-7B-Instruct",
                "device_map": "cuda",
                "attn_implementation": "flash_attention_2",
            },
            batch_size=4,
            limit=4 * num_gpus,  # Small limit for CI/CD
            num_gpus=num_gpus,
            timeout=600,
        )

        self.assertTrue(result.success, f"Evaluation failed: {result.error}")
        self.assertIsNotNone(result.job_result, "Job result should not be None")
        self.assertEqual(
            result.job_result.get("status"),
            "completed",
            f"Job status should be completed, got: {result.job_result.get('status')}",
        )


if __name__ == "__main__":
    unittest.main()

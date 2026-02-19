import json
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import utils
from utils import get_gpu_count, run_evaluation_test, with_server, with_temp_dir


class _FixedTemporaryDirectory:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _extract_strings(value):
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _extract_strings(item)


class TestReasoningTagStripping(TestCase):
    @with_temp_dir
    @with_server
    def test_reasoning_tag_stripping_qwen3vl(self, temp_dir, server):
        num_gpus = get_gpu_count()
        print(f"[TEST] Using {num_gpus} GPUs for evaluation")

        with patch.object(utils.tempfile, "TemporaryDirectory", return_value=_FixedTemporaryDirectory(temp_dir)):
            result = run_evaluation_test(
                server_url=server.url,
                model="qwen3_vl",
                tasks=["mme"],
                model_args={
                    "pretrained": "Qwen/Qwen3-VL-4B-Instruct",
                    "device_map": "cuda",
                    "attn_implementation": "flash_attention_2",
                },
                batch_size=1,
                limit=4 * num_gpus,
                num_gpus=num_gpus,
                timeout=1200,
            )

        self.assertTrue(result.success, f"Evaluation failed: {result.error}")
        self.assertIsNotNone(result.job_result, "Job result should not be None")
        self.assertEqual(
            result.job_result.get("status"),
            "completed",
            f"Job status should be completed, got: {result.job_result.get('status')}",
        )

        output_files = result.job_result.get("result") or {}
        self.assertTrue(output_files, "Job result should include output file metadata")

        sample_files = []
        for model_output in output_files.values():
            sample_files.extend(model_output.get("samples", []))

        self.assertTrue(sample_files, "No sample JSONL files found in output metadata")

        sample_rows = []
        for sample_file in sample_files:
            sample_path = Path(sample_file)
            self.assertTrue(sample_path.exists(), f"Sample file not found: {sample_file}")
            with sample_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        sample_rows.append(json.loads(line))

        self.assertTrue(sample_rows, "No sample rows found in JSONL outputs")

        raw_rows_with_think_tags = 0
        for row in sample_rows:
            self.assertIn("resps", row, "Sample row missing raw responses")
            self.assertIn("filtered_resps", row, "Sample row missing filtered responses")

            filtered_texts = list(_extract_strings(row["filtered_resps"]))
            self.assertTrue(filtered_texts, "filtered_resps should contain model output strings")
            for text in filtered_texts:
                self.assertNotIn("<think>", text)
                self.assertNotIn("</think>", text)

            raw_texts = list(_extract_strings(row["resps"]))
            self.assertTrue(raw_texts, "resps should preserve raw model outputs")
            if any("<think>" in text or "</think>" in text for text in raw_texts):
                raw_rows_with_think_tags += 1

        self.assertGreater(
            raw_rows_with_think_tags,
            0,
            "Expected at least one raw response in `resps` to contain <think> tags for Qwen3-VL",
        )


if __name__ == "__main__":
    unittest.main()

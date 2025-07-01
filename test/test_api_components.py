"""
Unit tests for core API components
"""
import unittest
from unittest.mock import Mock, patch

import pytest


class TestAPIComponents(unittest.TestCase):
    """Test core API components"""

    def test_instance_creation(self):
        """Test Instance class creation and properties"""
        from lmms_eval.api.instance import Instance

        # Test basic instance creation
        instance = Instance(
            request_type="generate_until",
            arguments=("test context", {"max_tokens": 100}),
            idx=0,
            metadata={"task": "test_task", "doc_id": 0, "repeats": 1},
        )

        self.assertEqual(instance.request_type, "generate_until")
        self.assertEqual(instance.idx, 0)
        self.assertEqual(instance.task_name, "test_task")
        self.assertEqual(instance.doc_id, 0)
        self.assertEqual(instance.repeats, 1)

    def test_model_registry(self):
        """Test model registration functionality"""
        from lmms_eval.api.registry import register_model

        # Test that decorator works
        @register_model("test_model")
        class TestModel:
            pass

        # Verify model was registered
        from lmms_eval.api.registry import MODEL_REGISTRY

        self.assertIn("test_model", MODEL_REGISTRY)
        self.assertEqual(MODEL_REGISTRY["test_model"], TestModel)

    def test_metrics_registry(self):
        """Test metrics registration functionality"""
        from lmms_eval.api.registry import register_metric

        # Test metric registration
        @register_metric(
            metric="test_metric",
            higher_is_better=True,
            output_type="generate_until",
            aggregation="mean",
        )
        def test_metric_fn(items):
            return items

        # Verify metric was registered
        from lmms_eval.api.registry import METRIC_REGISTRY

        self.assertIn("test_metric", METRIC_REGISTRY)

    def test_base_model_interface(self):
        """Test base model interface"""
        from lmms_eval.api.model import lmms

        # Create mock model
        class MockModel(lmms):
            def loglikelihood(self, requests):
                return [(0.5, True) for _ in requests]

            def generate_until(self, requests):
                return ["test response" for _ in requests]

            def generate_until_multi_round(self, requests):
                return ["test response" for _ in requests]

        model = MockModel()

        # Test interface methods exist
        self.assertTrue(hasattr(model, "loglikelihood"))
        self.assertTrue(hasattr(model, "generate_until"))
        self.assertTrue(hasattr(model, "generate_until_multi_round"))

        # Test properties
        self.assertEqual(model.rank, 0)
        self.assertEqual(model.world_size, 1)
        self.assertTrue(model.is_simple)

    def test_caching_functionality(self):
        """Test model caching functionality"""
        from lmms_eval.api.model import CacheHook, hash_args

        # Test hash function
        test_args = ("test", {"param": "value"})
        hash1 = hash_args("method", test_args)
        hash2 = hash_args("method", test_args)
        hash3 = hash_args("method", ("different", {"param": "value"}))

        self.assertEqual(hash1, hash2)  # Same inputs should hash the same
        self.assertNotEqual(hash1, hash3)  # Different inputs should hash differently

        # Test cache hook
        cache_hook = CacheHook(None)
        # Should not crash when dbdict is None
        cache_hook.add_partial("test_method", test_args, "result")

    @patch("lmms_eval.api.metrics.eval_logger")
    def test_metrics_calculation(self, mock_logger):
        """Test metrics calculation functions"""
        from lmms_eval.api.metrics import exact_match_hf_evaluate, mean, median

        # Test mean calculation
        test_values = [1, 2, 3, 4, 5]
        result = mean(test_values)
        self.assertEqual(result, 3.0)

        # Test median calculation
        result = median(test_values)
        self.assertEqual(result, 3)

        # Test exact match
        predictions = ["hello", "world", "test"]
        references = ["hello", "world", "different"]
        result = exact_match_hf_evaluate(predictions, references)
        expected_accuracy = 2 / 3  # 2 out of 3 matches
        self.assertAlmostEqual(result["exact_match"], expected_accuracy, places=3)

    def test_utility_functions(self):
        """Test utility functions"""
        from lmms_eval.utils import hash_string, simple_parse_args_string

        # Test argument parsing
        arg_string = "param1=value1,param2=value2,param3=123"
        parsed = simple_parse_args_string(arg_string)

        expected = {"param1": "value1", "param2": "value2", "param3": "123"}
        self.assertEqual(parsed, expected)

        # Test hash function
        test_string = "test string for hashing"
        hash1 = hash_string(test_string)
        hash2 = hash_string(test_string)
        hash3 = hash_string("different string")

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertIsInstance(hash1, str)


class TestTaskManagement(unittest.TestCase):
    """Test task management functionality"""

    def test_task_creation(self):
        """Test basic task creation and properties"""
        # This would require more complex setup with actual task files
        # For now, test that the task module can be imported
        try:
            from lmms_eval.api.task import Task
            from lmms_eval.tasks import TaskManager

            # Basic import test
            self.assertTrue(True, "Task modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import task modules: {e}")

    def test_task_registry(self):
        """Test task registration functionality"""
        from lmms_eval.api.registry import register_task

        # Test task registration decorator
        @register_task("test_task")
        class TestTask:
            pass

        from lmms_eval.api.registry import TASK_REGISTRY

        self.assertIn("test_task", TASK_REGISTRY)


if __name__ == "__main__":
    unittest.main()

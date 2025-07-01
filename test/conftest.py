"""
pytest configuration and fixtures for lmms-eval tests
"""
import os
import tempfile
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_model():
    """Mock model for testing without actual model loading"""
    mock = Mock()
    mock.generate.return_value = "test response"
    mock.tokenizer = Mock()
    mock.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock.tokenizer.decode.return_value = "test response"
    return mock


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_task_dict():
    """Mock task dictionary for testing"""
    return {
        "test_task": {
            "test": [
                {
                    "question": "What is 2+2?",
                    "answer": "4",
                    "image": None,
                    "doc_id": 0,
                }
            ]
        }
    }


@pytest.fixture
def mock_eval_logger():
    """Mock evaluation logger"""
    with patch("lmms_eval.api.model.eval_logger") as mock_logger:
        yield mock_logger

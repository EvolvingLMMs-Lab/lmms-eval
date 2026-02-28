"""Shared pytest fixtures for the lmms-eval test suite.

Provides reusable fixtures for temporary directories, mock models,
mock instances, and ChatMessages construction. All existing unittest-based
tests continue to work under pytest without modification.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from lmms_eval.api.instance import Instance

# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, yield its path, and clean up after."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Return a factory that creates a mock model returning given responses."""

    def _factory(responses, is_simple=False):
        m = MagicMock()
        m.generate_until = MagicMock(return_value=responses)
        m.loglikelihood = MagicMock(return_value=responses)
        m.is_simple = is_simple
        m.task_dict = {}
        return m

    return _factory


# ---------------------------------------------------------------------------
# Instance helper fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def make_instance():
    """Return a factory for creating Instance objects with sensible defaults."""

    def _factory(
        request_type="generate_until",
        prompt="test prompt",
        gen_kwargs=None,
        doc_id=0,
        idx=0,
        task="test_task",
        split="test",
        repeats=1,
    ):
        if gen_kwargs is None:
            gen_kwargs = {"temperature": 0, "until": ["\n"]}
        arguments = (prompt, gen_kwargs, None, doc_id, task, split)
        return Instance(
            request_type=request_type,
            arguments=arguments,
            idx=idx,
            metadata={"task": task, "doc_id": doc_id, "repeats": repeats},
        )

    return _factory


# ---------------------------------------------------------------------------
# ChatMessages helper fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def make_chat_messages():
    """Return a factory for creating ChatMessages from simple dicts."""
    from lmms_eval.protocol import ChatMessages

    def _factory(messages):
        """Build ChatMessages from a list of (role, content_list) tuples.

        Example:
            make_chat_messages([
                ("user", [("text", "What is this?"), ("image", some_image)]),
                ("assistant", [("text", "It's a cat.")]),
            ])
        """
        raw = []
        for role, content_items in messages:
            content = []
            for ctype, cvalue in content_items:
                if ctype == "text":
                    content.append({"type": "text", "text": cvalue})
                else:
                    content.append({"type": ctype, "url": cvalue})
            raw.append({"role": role, "content": content})
        return ChatMessages(messages=raw)

    return _factory


# ---------------------------------------------------------------------------
# Marker-based skips
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers and CLI options."""
    # Markers are already declared in pyproject.toml [tool.pytest.ini_options].
    # This hook is here for forward-compat if we add dynamic markers later.
    pass


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate golden snapshot files for prompt stability tests.",
    )


@pytest.fixture
def update_snapshots(request):
    """Whether to update (rather than compare) prompt stability snapshots."""
    return request.config.getoption("--update-snapshots")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU and API tests when the environment doesn't support them."""
    skip_gpu = pytest.mark.skip(reason="No GPU available")
    skip_api = pytest.mark.skip(reason="Required API key not set")

    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "api" in item.keywords:
            # Check common API key env vars
            has_key = any(os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"))
            if not has_key:
                item.add_marker(skip_api)

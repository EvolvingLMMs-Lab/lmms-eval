import importlib
import sys

import pytest

EVALUATORS = {
    "mathverse": ("lmms_eval.tasks.mathverse.mathverse_evals", "MathVerseEvaluator"),
    "mmrefine": ("lmms_eval.tasks.mmrefine.mmrefine_evals", "MMRefineEvaluator"),
}


@pytest.fixture
def judge_client(monkeypatch):
    """Build the evaluator's OpenAI client from the environment, as it is at import time."""
    loaded = []

    def load(task, **env):
        for name in ("OPENAI_API_URL", "OPENAI_API_BASE", "API_TYPE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        module_name, class_name = EVALUATORS[task]
        module = importlib.reload(importlib.import_module(module_name))
        loaded.append(module)
        return getattr(module, class_name).client

    yield load
    # Drop the modules built from the test environment so later imports start clean.
    for module in loaded:
        sys.modules.pop(module.__name__, None)


def chat_url(client):
    return str(client.base_url.join("chat/completions"))


@pytest.mark.parametrize("task", sorted(EVALUATORS))
@pytest.mark.parametrize(
    "base",
    [
        "https://api.deepseek.com",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://llm.example.com/openai",
        "http://localhost:8000/api",
        "https://api.openai.com/v1",
        "http://localhost:8000/v1",
    ],
)
def test_client_posts_to_the_configured_base(judge_client, task, base):
    client = judge_client(task, OPENAI_API_BASE=base)
    assert chat_url(client) == f"{base}/chat/completions"


@pytest.mark.parametrize("task", sorted(EVALUATORS))
def test_full_chat_completions_url_is_reduced_to_its_base(judge_client, task):
    client = judge_client(task, OPENAI_API_URL="https://llm.example.com/openai/chat/completions")
    assert chat_url(client) == "https://llm.example.com/openai/chat/completions"


@pytest.mark.parametrize("task", sorted(EVALUATORS))
def test_default_endpoint_is_openai(judge_client, task):
    client = judge_client(task)
    assert chat_url(client) == "https://api.openai.com/v1/chat/completions"

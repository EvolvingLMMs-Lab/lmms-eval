from lmms_eval.llm_judge.openai_endpoint import (
    DEFAULT_OPENAI_COMPAT_BASE,
    resolve_openai_api_base,
    resolve_openai_chat_completions_url,
    resolve_openai_model_base,
)


def test_default_is_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    assert resolve_openai_api_base() == DEFAULT_OPENAI_COMPAT_BASE
    assert resolve_openai_chat_completions_url() == f"{DEFAULT_OPENAI_COMPAT_BASE}/chat/completions"


def test_url_wins_over_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_BASE", "https://ignored.test/v1")
    assert resolve_openai_api_base() == "https://example.test/v1"


def test_base_alias_and_chat_suffix_stripped(monkeypatch):
    monkeypatch.delenv("OPENAI_API_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    assert resolve_openai_api_base() == "https://api.openai.com/v1"
    assert resolve_openai_chat_completions_url() == "https://api.openai.com/v1/chat/completions"


def test_local_model_and_remote_judge_are_independent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_API_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ignored.test/v1")
    assert resolve_openai_model_base() == "http://localhost:8000/v1"
    assert resolve_openai_api_base() == "https://api.openai.com/v1"
    assert resolve_openai_chat_completions_url() == "https://api.openai.com/v1/chat/completions"


def test_model_defers_standard_base_url_to_sdk(monkeypatch):
    from openai import OpenAI

    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_URL", "https://judge.test/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert resolve_openai_model_base() is None
    with OpenAI(api_key="test-only", base_url=resolve_openai_model_base()) as client:
        assert str(client.base_url) == "http://localhost:8000/v1/"


def test_model_default_is_official_openai(monkeypatch):
    from openai import OpenAI

    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_URL", "https://judge.test/v1")
    with OpenAI(api_key="test-only", base_url=resolve_openai_model_base()) as client:
        assert str(client.base_url) == "https://api.openai.com/v1/"

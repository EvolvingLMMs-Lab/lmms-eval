from lmms_eval.llm_judge.openai_endpoint import (
    DEFAULT_OPENAI_COMPAT_BASE,
    resolve_openai_api_base,
    resolve_openai_chat_completions_url,
)


def test_default_is_openrouter(monkeypatch):
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

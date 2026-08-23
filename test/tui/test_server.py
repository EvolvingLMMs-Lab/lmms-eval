import asyncio

import httpx
import pytest

from lmms_eval.tui import server


def _request(
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://tui.test") as client:
            return await client.request(method, path, headers=headers, json=json)

    return asyncio.run(send())


def test_default_tui_origins_are_exact_local_origins(monkeypatch):
    monkeypatch.delenv("LMMS_EVAL_TUI_ALLOWED_ORIGINS", raising=False)

    assert server._allowed_tui_origins() == (
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )


@pytest.mark.parametrize("value", ["*", "https://tui.example/path", "https://user:pass@tui.example"])
def test_invalid_tui_origin_override_fails_closed(monkeypatch, value):
    monkeypatch.setenv("LMMS_EVAL_TUI_ALLOWED_ORIGINS", value)

    with pytest.raises(RuntimeError, match="LMMS_EVAL_TUI_ALLOWED_ORIGINS"):
        server._allowed_tui_origins()


def test_cors_allows_only_configured_origin():
    allowed = _request(
        "OPTIONS",
        "/eval/preview",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    denied = _request(
        "OPTIONS",
        "/eval/preview",
        headers={"Origin": "https://attacker.invalid", "Access-Control-Request-Method": "POST"},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in allowed.headers
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_preview_and_start_reject_request_environment_text():
    payload = {
        "model": "dummy",
        "tasks": ["mme"],
        "env_setup": "source .venv/bin/activate; touch /tmp/pwned",
        "env_vars": "OPENAI_API_KEY=$(cat ~/.ssh/id_rsa)",
    }

    for endpoint in ("/eval/preview", "/eval/start", "/eval/export-yaml"):
        response = _request("POST", endpoint, json=payload)
        assert response.status_code == 400
        assert response.json()["detail"] == "Request-level environment setup is disabled; configure the TUI server environment instead."


def test_preview_quotes_hostile_values_without_shell_prefix():
    request = server.PreviewRequest(
        model="demo; touch /tmp/model",
        model_args="token=$(touch /tmp/args)",
        tasks=["mme; touch /tmp/task", "task with spaces"],
        output_path="../logs; touch /tmp/path",
        device="cuda:0; touch /tmp/device",
    )

    preview = server._build_command(request)

    assert preview.startswith("python -m lmms_eval ")
    assert "source " not in preview
    assert "export " not in preview
    assert "'demo; touch /tmp/model'" in preview
    assert "'../logs; touch /tmp/path'" in preview


def test_stream_launches_hostile_values_as_structured_argv(monkeypatch):
    request = server.EvalRequest(
        model="demo; touch /tmp/model",
        model_args="token=$(touch /tmp/args)",
        tasks=["mme; touch /tmp/task", "task with spaces"],
        output_path="../logs; touch /tmp/path",
        device="cuda:0; touch /tmp/device",
    )
    job_id = "structured-argv-test"
    server._jobs[job_id] = {
        "status": "starting",
        "argv": server._build_eval_argv(request),
        "process": None,
        "request": request,
    }
    captured: dict[str, object] = {}

    class FakeProcess:
        stdout = None
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    async def shell_must_not_run(*args, **kwargs):
        raise AssertionError("create_subprocess_shell must not be called")

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(server.asyncio, "create_subprocess_shell", shell_must_not_run)

    async def consume() -> list[str]:
        return [event async for event in server._stream_output(job_id)]

    events = asyncio.run(consume())

    assert any('"type": "done"' in event for event in events)
    assert captured["argv"] == tuple(server._build_eval_argv(request))
    assert captured["kwargs"] == {
        "stdout": server.asyncio.subprocess.PIPE,
        "stderr": server.asyncio.subprocess.STDOUT,
        "start_new_session": True,
        "cwd": server.get_repo_root() or None,
    }

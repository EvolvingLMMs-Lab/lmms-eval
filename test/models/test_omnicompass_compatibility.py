"""Default backend behavior stays compatible; strict evaluation is opt-in."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from PIL import Image

from lmms_eval import __main__ as cli
from lmms_eval.models.chat.openai import OpenAICompatible
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL
from lmms_eval.tasks import TaskManager
from test.models import test_qwen3_vl_simple as qwen_fixtures


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("first_status", [200, 400, 429, 503])
def test_api_retry_and_timeout_defaults_are_preserved(strict, first_status):
    records = []

    def respond(request):
        status = first_status if not records else 200
        records.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "test failure"}})
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "B"}, "finish_reason": "stop"}],
            },
        )

    model = OpenAICompatible(
        api_key="test-placeholder",
        base_url="https://example.invalid/v1",
        strict_generation=strict,
        max_retries=2 if first_status == 400 else 1,
        num_concurrent=1,
        prefix_aware_queue=False,
        retry_backoff_s=0,
    )
    original = model.client
    baseline_timeout = original.timeout
    model.client = original.with_options(http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    model.task_dict = {"mme": {"test": [{}]}}
    request = SimpleNamespace(args=("Question", lambda doc: [{"role": "user", "content": [{"type": "text", "text": "Choose A or B."}]}], {"temperature": 0, "max_new_tokens": 128}, 0, "mme", "test"))
    try:
        with patch("time.sleep"), patch("lmms_eval.models.chat.openai.is_budget_exceeded", return_value=False):
            result = model.generate_until([request])[0]
        assert model.client.max_retries == (0 if strict else 2)
        assert len(records) == (1 if strict or first_status == 200 else 2)
        assert records[0].extensions["timeout"]["read"] == (model.timeout if strict else baseline_timeout.read)
        assert json.loads(records[0].content)["max_tokens"] == 128
        assert (result.text == "B") == (not strict or first_status == 200)
    finally:
        model.client.close()
        original.close()


@pytest.mark.parametrize("pixel_limits", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_qwen_input_defaults_and_strict_context_are_independent(pixel_limits, strict):
    fixture = qwen_fixtures.TestQwen3VLSimple()
    model = object.__new__(Qwen3_VL)
    model.__dict__.update(fixture._make_model().__dict__)
    model.apply_image_pixel_limits = pixel_limits
    model.strict_generation = strict
    model._model.config = SimpleNamespace(max_position_embeddings=64)
    model._model.generate = Mock(wraps=model._model.generate)
    messages = []

    def vision(batch, **kwargs):
        messages.extend(batch)
        return [Image.new("RGB", (64, 64))], None, {}

    request = SimpleNamespace(
        args=(
            "Question",
            lambda doc: [{"role": "user", "content": [{"type": "image", "url": Image.new("RGB", (64, 64))}]}],
            {"max_new_tokens": 128},
            0,
            "demo_task",
            "test",
        )
    )
    with patch("lmms_eval.models.chat.qwen3_vl.process_vision_info", vision):
        if strict:
            with pytest.raises(ValueError, match="Context capacity exceeded"):
                model.generate_until([request])
            model._model.generate.assert_not_called()
        else:
            result = model.generate_until([request])[0]
            assert result.text == "final answer"
            assert result.token_counts.input_tokens is None
            model._model.generate.assert_called_once()
    image = messages[0][1]["content"][0]
    assert ("min_pixels" in image) == pixel_limits
    assert ("max_pixels" in image) == pixel_limits
    if pixel_limits:
        assert (image["min_pixels"], image["max_pixels"]) == (model.min_pixels, model.max_pixels)


@pytest.mark.parametrize("strict", [False, True])
def test_cli_error_exit_is_explicit_opt_in(monkeypatch, strict):
    argv = ["lmms_eval", "--model", "openai", "--tasks", "mme"]
    if strict:
        argv.append("--fail_on_error")
    monkeypatch.setattr("sys.argv", argv)
    accelerator = Mock()
    accelerator.is_main_process = True
    with patch.object(cli, "Accelerator", return_value=accelerator), patch.object(cli, "cli_evaluate_single", side_effect=ValueError("test failure")):
        if strict:
            with pytest.raises(SystemExit) as exc:
                cli.cli_evaluate()
            assert exc.value.code == 1
        else:
            cli.cli_evaluate()


def test_existing_python_task_model_name_is_not_changed():
    class LegacyModelAwareTask:
        def __init__(self, config=None, model_name=None):
            self.model_name = model_name

    tasks = TaskManager(model_name="openai").load_task_or_group([{"task": "existing_task", "class": LegacyModelAwareTask}])
    assert tasks["existing_task"].model_name is None

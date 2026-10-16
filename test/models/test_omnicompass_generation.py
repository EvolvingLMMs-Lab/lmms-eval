"""Generation-contract tests with fake weights; these do not validate a GPU."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from lmms_eval import __main__ as cli
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
from lmms_eval.protocol import ChatMessages


def qwen(thinking: bool = True, context_length: int | None = None) -> Qwen3_VL:
    model = object.__new__(Qwen3_VL)
    model._model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=65536)))
    model.enable_thinking = thinking
    model.context_length = context_length
    model.return_response_json = True
    return model


@pytest.mark.parametrize("thinking", [True, False])
def test_context_check_keeps_requested_budget(thinking: bool) -> None:
    model = qwen(thinking)
    model._check_context({"input_ids": torch.zeros(1, 32768)}, 32768)
    with pytest.raises(ValueError, match="Context capacity exceeded"):
        model._check_context({"input_ids": torch.zeros(1, 32769)}, 32768)
    assert model.context_length is None


def test_explicit_context_cannot_override_smaller_model_capacity() -> None:
    with pytest.raises(ValueError, match="65536 total"):
        qwen(context_length=131072)._check_context({"input_ids": torch.zeros(1, 40000)}, 32768)
    with pytest.raises(ValueError, match="32768 total"):
        qwen(context_length=32768)._check_context({"input_ids": torch.zeros(1, 1)}, 32768)


@pytest.mark.parametrize(
    ("thinking", "raw", "final"),
    [(True, "<think>reasoning</think>C", "C"), (True, "<think>unfinished", ""), (False, "C", "C")],
)
@pytest.mark.parametrize("output_tokens", [16, 32768])
def test_raw_thinking_and_finish_reason_retained(thinking: bool, raw: str, final: str, output_tokens: int) -> None:
    result = json.loads(qwen(thinking)._format_response(raw, 128, output_tokens, 32768))
    body = result["api_response"]["body"]
    choice = body["choices"][0]
    assert choice["message"] == {"content": final, "raw_generation": raw}
    assert choice["finish_reason"] == ("length" if output_tokens == 32768 else "stop")
    assert body["usage"] == {"prompt_tokens": 128, "completion_tokens": output_tokens}


@pytest.mark.parametrize("encoding", ["base64", "data-uri"])
@pytest.mark.parametrize(("extension", "mime"), [("mp3", "audio/mpeg"), ("wav", "audio/wav")])
def test_local_mp3_is_encoded_and_message_order_preserved(tmp_path: Path, encoding: str, extension: str, mime: str) -> None:
    audio = tmp_path / f"audio.{extension}"
    audio.write_bytes(b"fake-mp3")
    messages = ChatMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Timestamp: 0 seconds."},
                    {"type": "audio", "url": str(audio)},
                    {"type": "text", "text": "Question"},
                ],
            }
        ]
    )
    content = messages.to_openai_messages(audio_format="input_audio", audio_encoding=encoding)[0]["content"]
    assert [item["type"] for item in content] == ["text", "input_audio", "text"]
    encoded = base64.b64encode(b"fake-mp3").decode("ascii")
    assert content[1]["input_audio"] == {"data": (f"data:{mime};base64," if encoding == "data-uri" else "") + encoded, "format": extension}


@pytest.mark.parametrize("task", ["omnicompass_vl", "mme"])
def test_native_cli_returns_nonzero_on_evaluation_failure(monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    accelerator = Mock()
    accelerator.is_main_process = True
    monkeypatch.setattr("sys.argv", ["lmms_eval", "--model", "openai", "--tasks", task])
    with patch.object(cli, "Accelerator", return_value=accelerator), patch.object(cli, "cli_evaluate_single", side_effect=ValueError("test context overflow")):
        with pytest.raises(SystemExit) as error:
            cli.cli_evaluate()
    assert error.value.code == 1
    accelerator.state.destroy_process_group.assert_called_once()

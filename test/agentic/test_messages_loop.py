"""Stateful message episodes use the normal backend without caching live state."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.evaluator import _run_generate_until_agentic


def _request(callback, max_steps=2, stateful=True):
    return Instance(
        request_type="generate_until_agentic",
        arguments=("context", callback, {"max_agentic_steps": max_steps}, 0, "game", "test"),
        idx=0,
        metadata={"task": "game", "doc_id": 0, "repeats": 1, "stateful": stateful},
    )


def _model():
    model = SimpleNamespace(task_dict={"game": {"test": [{}]}}, is_simple=False)
    model.generate_until = Mock(return_value=["FORWARD"])
    return model


def test_live_messages_advance_close_and_bypass_response_cache():
    closed = Mock()
    calls = []
    frame = Image.new("RGB", (2, 2))

    def callback(doc, round_idx=None, previous_output=None, previous_round_info=None):
        calls.append(round_idx)
        info = {"close": closed, "state": {"steps": round_idx}, "finalize": lambda: json.dumps({"success": False, "metrics": {"steps": round_idx}})}
        messages = [{"role": "user", "content": [{"type": "image", "url": frame}, {"type": "text", "text": str(round_idx)}]}]
        return messages, False, previous_output, info

    lm = _model()
    cache = Mock()
    [response] = _run_generate_until_agentic(lm, [_request(callback)], agentic_trace_mode="full", response_cache=cache)

    assert calls == [0, 1, 2]
    assert lm.generate_until.call_count == 2
    cache.execute.assert_not_called()
    closed.assert_called_once()
    assert json.loads(response)["metrics"]["steps"] == 2
    assert [step["state"]["steps"] for step in json.loads(response)["agentic_rounds"]] == [1, 2]
    inner = lm.generate_until.call_args.args[0][0]
    assert inner.args[1]({})[0]["content"][0]["url"] is frame


@pytest.mark.parametrize("failure", ["backend", "callback"])
def test_episode_errors_close_environment_and_do_not_abort_following_samples(failure):
    closed = Mock()
    info = {"close": closed}

    def callback(doc, round_idx=None, previous_output=None, previous_round_info=None):
        if round_idx and failure == "callback":
            raise RuntimeError("step failed")
        return [{"role": "user", "content": [{"type": "text", "text": "move"}]}], False, previous_output, info

    lm = _model()
    if failure == "backend":
        lm.generate_until.side_effect = RuntimeError("endpoint failed")
    responses = _run_generate_until_agentic(lm, [_request(callback), _request(callback)])
    assert len(responses) == 2
    assert all(json.loads(response)["metrics"]["env_error"] == 1.0 for response in responses)
    assert closed.call_count == 2


def test_terminal_task_metrics_win_over_raw_model_output():
    closed = Mock()

    def callback(doc, round_idx=None, previous_output=None, previous_round_info=None):
        info = {"close": closed}
        if round_idx:
            return None, True, [*previous_output, '{"success":true,"metrics":{"steps":1}}'], info
        return [{"role": "user", "content": [{"type": "text", "text": "move"}]}], False, [], info

    lm = _model()
    [response] = _run_generate_until_agentic(lm, [_request(callback)])
    assert json.loads(response)["success"] is True
    assert lm.generate_until.call_count == 1
    closed.assert_called_once()

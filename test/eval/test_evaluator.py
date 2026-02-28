"""Unit tests for lmms_eval/evaluator.py — _run_generate_until_agentic.

Covers: single-step terminal, multi-step, max-steps fallback,
cache integration, is_simple vs chat model paths.

Does NOT cover: simple_evaluate, evaluate (too many dependencies).
"""

import copy
import json
from unittest.mock import MagicMock

import pytest

from lmms_eval.api.instance import Instance
from lmms_eval.evaluator import _run_generate_until_agentic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _terminal_doc_to_text(doc, previous_output, round_idx, previous_round_info):
    """Doc_to_text that terminates on first round."""
    return (
        None,  # visuals
        None,  # next_context
        True,  # terminal signal
        previous_output,  # updated_outputs
        {
            "state": {"round": round_idx},
            "tool_calls": 0,
            "valid_tool_calls": 0,
            "invalid_steps": 0,
            "prev": previous_round_info,
        },
    )


def _multi_step_doc_to_text(terminate_at=2):
    """Return a doc_to_text that terminates after `terminate_at` rounds."""

    def _inner(doc, previous_output, round_idx, previous_round_info):
        if round_idx >= terminate_at:
            return (
                None,
                None,
                True,  # terminal
                previous_output,
                {"state": {"done": True}, "tool_calls": round_idx, "valid_tool_calls": round_idx, "invalid_steps": 0},
            )
        return (
            None,  # visuals
            f"next_context_round_{round_idx + 1}",  # next_context
            False,  # not terminal
            None,  # don't update outputs
            {"state": {"round": round_idx}, "tool_calls": round_idx, "valid_tool_calls": round_idx, "invalid_steps": 0},
        )

    return _inner


def _never_terminal_doc_to_text(doc, previous_output, round_idx, previous_round_info):
    """Doc_to_text that never terminates — forces max_agentic_steps."""
    return (
        None,
        f"keep_going_{round_idx}",
        False,
        None,
        {"state": {"round": round_idx}, "tool_calls": round_idx, "valid_tool_calls": round_idx, "invalid_steps": 0},
    )


def _make_simple_model(responses):
    """Create a simple model (is_simple=True) that returns given responses one by one."""
    m = MagicMock()
    m.is_simple = True
    m.task_dict = {"agentic_task": {"test": [{"id": i} for i in range(100)]}}
    call_idx = [0]

    def _gen(requests):
        results = []
        for _ in requests:
            if call_idx[0] < len(responses):
                results.append(responses[call_idx[0]])
                call_idx[0] += 1
            else:
                results.append("default_response")
        return results

    m.generate_until = MagicMock(side_effect=_gen)
    return m


def _make_agentic_request(prompt="initial_prompt", doc_id=0, max_steps=3, doc_to_text_fn=None):
    """Create an Instance for generate_until_agentic."""
    if doc_to_text_fn is None:
        doc_to_text_fn = _terminal_doc_to_text
    gen_kwargs = {"temperature": 0, "max_agentic_steps": max_steps}
    return Instance(
        request_type="generate_until_agentic",
        arguments=(
            prompt,
            gen_kwargs,
            lambda _doc: [],  # doc_to_visual
            doc_to_text_fn,
            doc_id,
            "agentic_task",
            "test",
        ),
        idx=0,
        metadata={"task": "agentic_task", "doc_id": doc_id, "repeats": 1},
    )


# ===========================================================================
# Single-step terminal
# ===========================================================================


def test_single_step_terminal_returns_model_output_in_payload():
    """Single-step terminal fires on round 1, output wrapped in fallback payload."""
    lm = _make_simple_model(["answer_1"])
    req = _make_agentic_request(doc_to_text_fn=_terminal_doc_to_text)
    results = _run_generate_until_agentic(lm, [req])
    assert len(results) == 1
    # Terminal fires on round 1, but post-loop wraps non-JSON output in fallback payload
    payload = json.loads(results[0])
    assert payload["last_model_output"] == "answer_1"
    assert "trace" in payload


def test_single_step_terminal_model_called_once():
    """Single-step terminal calls model exactly once."""
    lm = _make_simple_model(["answer"])
    req = _make_agentic_request(doc_to_text_fn=_terminal_doc_to_text)
    _run_generate_until_agentic(lm, [req])
    assert lm.generate_until.call_count == 1


def test_single_step_terminal_multiple_requests():
    """Single-step terminal handles multiple requests correctly."""
    lm = _make_simple_model(["a1", "a2"])
    reqs = [
        _make_agentic_request(prompt="p1", doc_id=0, doc_to_text_fn=_terminal_doc_to_text),
        _make_agentic_request(prompt="p2", doc_id=1, doc_to_text_fn=_terminal_doc_to_text),
    ]
    results = _run_generate_until_agentic(lm, reqs)
    assert len(results) == 2


# ===========================================================================
# Multi-step
# ===========================================================================


def test_multi_step_two_rounds_then_terminal():
    """Multi-step evaluation terminates after two rounds."""
    lm = _make_simple_model(["round1_output", "round2_output"])
    req = _make_agentic_request(max_steps=5, doc_to_text_fn=_multi_step_doc_to_text(terminate_at=2))
    results = _run_generate_until_agentic(lm, [req])
    assert len(results) == 1
    # Model called twice (round 1 and round 2)
    assert lm.generate_until.call_count == 2


def test_multi_step_three_rounds():
    """Multi-step evaluation runs three rounds."""
    lm = _make_simple_model(["r1", "r2", "r3"])
    req = _make_agentic_request(max_steps=5, doc_to_text_fn=_multi_step_doc_to_text(terminate_at=3))
    _run_generate_until_agentic(lm, [req])
    assert lm.generate_until.call_count == 3


# ===========================================================================
# Max steps reached — fallback JSON
# ===========================================================================


def test_max_steps_reached_fallback_json_on_max_steps():
    """Max steps reached generates fallback JSON with error marker."""
    lm = _make_simple_model(["step1", "step2", "step3"])
    req = _make_agentic_request(max_steps=2, doc_to_text_fn=_never_terminal_doc_to_text)
    results = _run_generate_until_agentic(lm, [req])
    assert len(results) == 1
    # When max steps reached and last output isn't JSON, a fallback JSON is generated
    payload = json.loads(results[0])
    assert payload["error"] == "max_agentic_steps_reached"
    assert payload["success"] is False
    assert "last_model_output" in payload
    assert "trace" in payload


def test_max_steps_reached_fallback_has_trace():
    """Max steps reached fallback includes trace of all rounds."""
    lm = _make_simple_model(["s1", "s2"])
    req = _make_agentic_request(max_steps=2, doc_to_text_fn=_never_terminal_doc_to_text)
    results = _run_generate_until_agentic(lm, [req])
    payload = json.loads(results[0])
    assert isinstance(payload["trace"], list)
    assert len(payload["trace"]) == 2


def test_max_steps_reached_model_called_max_steps_times():
    """Max steps reached calls model exactly max_steps times."""
    lm = _make_simple_model(["s1", "s2", "s3"])
    req = _make_agentic_request(max_steps=3, doc_to_text_fn=_never_terminal_doc_to_text)
    _run_generate_until_agentic(lm, [req])
    assert lm.generate_until.call_count == 3


# ===========================================================================
# Cache integration
# ===========================================================================


def test_cache_integration_cache_is_used_when_provided():
    """Cache integration uses cache when provided."""
    lm = _make_simple_model(["cached_answer"])
    cache = MagicMock()
    cache.execute = MagicMock(return_value=["cached_answer"])

    req = _make_agentic_request(doc_to_text_fn=_terminal_doc_to_text)
    results = _run_generate_until_agentic(lm, [req], response_cache=cache)

    assert len(results) == 1
    # Cache.execute should be called instead of direct model.generate_until
    cache.execute.assert_called()
    # Direct generate_until should NOT be called when cache is used
    lm.generate_until.assert_not_called()


def test_cache_integration_no_cache_calls_model_directly():
    """Cache integration calls model directly when no cache provided."""
    lm = _make_simple_model(["direct_answer"])
    req = _make_agentic_request(doc_to_text_fn=_terminal_doc_to_text)
    _run_generate_until_agentic(lm, [req], response_cache=None)
    lm.generate_until.assert_called()


# ===========================================================================
# is_simple model path
# ===========================================================================


def test_simple_model_path_creates_generate_until_request():
    """is_simple=True path creates Instance with correct args structure."""
    lm = _make_simple_model(["simple_out"])
    req = _make_agentic_request(doc_to_text_fn=_terminal_doc_to_text)
    _run_generate_until_agentic(lm, [req])

    # Check the Instance passed to generate_until
    call_args = lm.generate_until.call_args[0][0]
    assert len(call_args) == 1
    inner_req = call_args[0]
    assert inner_req.request_type == "generate_until"
    # Simple path: (ctx, gen_kwargs, doc_to_visual, doc_id, task, split) — 6 elements
    assert len(inner_req.args) == 6
    # gen_kwargs should not have max_agentic_steps (it gets popped)
    assert "max_agentic_steps" not in inner_req.args[1]


# ===========================================================================
# doc_to_text returning plain string
# ===========================================================================


def test_doc_to_text_string_return_updates_context():
    """When doc_to_text returns a plain string, it becomes the next context."""
    call_count = [0]

    def _string_doc_to_text(doc, previous_output, round_idx, previous_round_info):
        call_count[0] += 1
        if round_idx >= 2:
            return (None, None, True, previous_output, {"state": {}, "tool_calls": 0, "valid_tool_calls": 0, "invalid_steps": 0})
        return f"updated_context_{round_idx}"

    lm = _make_simple_model(["r1", "r2"])
    req = _make_agentic_request(max_steps=5, doc_to_text_fn=_string_doc_to_text)
    results = _run_generate_until_agentic(lm, [req])
    assert len(results) == 1


# ===========================================================================
# Instance args structure
# ===========================================================================


def test_agentic_instance_args_has_7_elements():
    """Agentic instance has exactly 7 elements in args tuple."""
    req = _make_agentic_request()
    assert len(req.args) == 7


def test_agentic_instance_args_unpacks_correctly():
    """Agentic instance args unpack to correct types and values."""
    req = _make_agentic_request(prompt="test", doc_id=5)
    ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task_name, split = req.args
    assert ctx == "test"
    assert isinstance(gen_kwargs, dict)
    assert callable(doc_to_visual)
    assert callable(doc_to_text)
    assert doc_id == 5
    assert task_name == "agentic_task"
    assert split == "test"


def test_agentic_instance_args_gen_kwargs_has_max_agentic_steps():
    """Agentic instance gen_kwargs includes max_agentic_steps."""
    req = _make_agentic_request(max_steps=7)
    assert req.args[1]["max_agentic_steps"] == 7

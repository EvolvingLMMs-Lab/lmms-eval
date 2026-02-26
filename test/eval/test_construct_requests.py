"""Unit tests for construct_requests tuple shapes.

Validates the critical req.args contract that models depend on to unpack
Instance arguments. Does NOT instantiate real ConfigurableTask (which
requires dataset download). Instead, we manually construct Instance objects
with the same tuple shapes and verify unpacking works correctly.

The documented contract (from AGENTS.md):
  ConfigurableTask generate_until:      (ctx, gen_kwargs, doc_to_visual, doc_id, task, split) — 6 elements
  ConfigurableTask loglikelihood:        (ctx, doc_to_target, doc_to_visual, doc_id, task, split) — 6 elements
  ConfigurableTask generate_until_multi_round: (ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split) — 7 elements
  ConfigurableTask generate_until_agentic:     (ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split) — 7 elements
  ConfigurableMessagesTask generate_until:     (ctx, doc_to_messages, gen_kwargs, doc_id, task, split) — 6 elements
  ConfigurableMessagesTask generate_until_agentic: (ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split) — 7 elements
"""

import copy
from functools import partial

import pytest

from lmms_eval.api.instance import Instance

# ---------------------------------------------------------------------------
# Helpers: mirror what construct_requests produces
# ---------------------------------------------------------------------------


def _dummy_doc_to_visual(doc):
    return []


def _dummy_doc_to_target(doc):
    return "target text"


def _dummy_doc_to_messages(doc):
    return [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def _dummy_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return "prompt text"


_GEN_KWARGS = {"temperature": 0, "until": ["\n"], "max_new_tokens": 512}


def _make_instance(request_type, arguments, doc_id=0, task="test_task", split="test"):
    return Instance(
        request_type=request_type,
        arguments=arguments,
        idx=0,
        metadata={"task": task, "doc_id": doc_id, "repeats": 1, "split": split},
    )


# ---------------------------------------------------------------------------
# ConfigurableTask generate_until tests
# ---------------------------------------------------------------------------


def test_configurable_task_generate_until_tuple_length_is_6():
    """ConfigurableTask generate_until produces 6-element tuple."""
    # Arrange
    args = ("prompt", copy.deepcopy(_GEN_KWARGS), _dummy_doc_to_visual, 0, "mme", "test")
    inst = _make_instance("generate_until", args, doc_id=0, task="mme", split="test")

    # Act & Assert
    assert len(inst.args) == 6


def test_configurable_task_generate_until_unpack_order():
    """ConfigurableTask generate_until unpacks in correct order."""
    # Arrange
    args = ("my prompt", copy.deepcopy(_GEN_KWARGS), _dummy_doc_to_visual, 42, "mme", "val")
    inst = _make_instance("generate_until", args, doc_id=42, task="mme", split="val")

    # Act
    ctx, gen_kwargs, doc_to_visual, doc_id, task, split = inst.args

    # Assert
    assert ctx == "my prompt"
    assert isinstance(gen_kwargs, dict)
    assert "temperature" in gen_kwargs
    assert callable(doc_to_visual)
    assert doc_id == 42
    assert task == "mme"
    assert split == "val"


def test_configurable_task_generate_until_gen_kwargs_is_dict():
    """ConfigurableTask generate_until has dict at index 1."""
    # Arrange
    args = ("prompt", copy.deepcopy(_GEN_KWARGS), _dummy_doc_to_visual, 0, "mme", "test")
    inst = _make_instance("generate_until", args)

    # Act & Assert
    assert isinstance(inst.args[1], dict)


def test_configurable_task_generate_until_doc_to_visual_is_callable():
    """ConfigurableTask generate_until has callable at index 2."""
    # Arrange
    args = ("prompt", copy.deepcopy(_GEN_KWARGS), _dummy_doc_to_visual, 0, "mme", "test")
    inst = _make_instance("generate_until", args)

    # Act & Assert
    assert callable(inst.args[2])


# ---------------------------------------------------------------------------
# ConfigurableTask loglikelihood tests
# ---------------------------------------------------------------------------


def test_configurable_task_loglikelihood_tuple_length_is_6():
    """ConfigurableTask loglikelihood produces 6-element tuple."""
    # Arrange
    args = ("context", _dummy_doc_to_target, _dummy_doc_to_visual, 0, "mmlu", "test")
    inst = _make_instance("loglikelihood", args, doc_id=0, task="mmlu", split="test")

    # Act & Assert
    assert len(inst.args) == 6


def test_configurable_task_loglikelihood_unpack_order():
    """ConfigurableTask loglikelihood unpacks in correct order."""
    # Arrange
    args = ("What is 2+2?", _dummy_doc_to_target, _dummy_doc_to_visual, 7, "mmlu", "val")
    inst = _make_instance("loglikelihood", args, doc_id=7, task="mmlu", split="val")

    # Act
    ctx, doc_to_target, doc_to_visual, doc_id, task, split = inst.args

    # Assert
    assert ctx == "What is 2+2?"
    assert callable(doc_to_target)
    assert callable(doc_to_visual)
    assert doc_id == 7
    assert task == "mmlu"
    assert split == "val"


def test_configurable_task_loglikelihood_doc_to_target_is_callable():
    """ConfigurableTask loglikelihood has callable at index 1."""
    # Arrange
    args = ("context", _dummy_doc_to_target, _dummy_doc_to_visual, 0, "mmlu", "test")
    inst = _make_instance("loglikelihood", args)

    # Act & Assert
    assert callable(inst.args[1])


# ---------------------------------------------------------------------------
# ConfigurableTask multi-round tests
# ---------------------------------------------------------------------------


def test_configurable_task_multi_round_tuple_length_is_7():
    """ConfigurableTask generate_until_multi_round produces 7-element tuple."""
    # Arrange
    args = (
        "round1",
        copy.deepcopy(_GEN_KWARGS),
        _dummy_doc_to_visual,
        partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
        0,
        "multi",
        "test",
    )
    inst = _make_instance("generate_until_multi_round", args, doc_id=0, task="multi", split="test")

    # Act & Assert
    assert len(inst.args) == 7


def test_configurable_task_multi_round_unpack_order():
    """ConfigurableTask generate_until_multi_round unpacks in correct order."""
    # Arrange
    args = (
        "round1",
        copy.deepcopy(_GEN_KWARGS),
        _dummy_doc_to_visual,
        partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
        3,
        "multi",
        "val",
    )
    inst = _make_instance("generate_until_multi_round", args, doc_id=3, task="multi", split="val")

    # Act
    ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split = inst.args

    # Assert
    assert ctx == "round1"
    assert isinstance(gen_kwargs, dict)
    assert callable(doc_to_visual)
    assert callable(doc_to_text)
    assert doc_id == 3
    assert task == "multi"
    assert split == "val"


# ---------------------------------------------------------------------------
# ConfigurableTask agentic tests
# ---------------------------------------------------------------------------


def test_configurable_task_agentic_tuple_length_is_7():
    """ConfigurableTask generate_until_agentic produces 7-element tuple."""
    # Arrange
    gk = copy.deepcopy(_GEN_KWARGS)
    gk["max_agentic_steps"] = 5
    args = (
        "agentic_prompt",
        gk,
        _dummy_doc_to_visual,
        partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
        0,
        "agent",
        "test",
    )
    inst = _make_instance("generate_until_agentic", args, doc_id=0, task="agent", split="test")

    # Act & Assert
    assert len(inst.args) == 7


def test_configurable_task_agentic_has_max_agentic_steps():
    """ConfigurableTask generate_until_agentic gen_kwargs contains max_agentic_steps."""
    # Arrange
    gk = copy.deepcopy(_GEN_KWARGS)
    gk["max_agentic_steps"] = 5
    args = (
        "agentic_prompt",
        gk,
        _dummy_doc_to_visual,
        partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
        0,
        "agent",
        "test",
    )
    inst = _make_instance("generate_until_agentic", args)

    # Act
    _, gen_kwargs, *_ = inst.args

    # Assert
    assert gen_kwargs["max_agentic_steps"] == 5


def test_configurable_task_agentic_unpack_order():
    """ConfigurableTask generate_until_agentic unpacks in correct order."""
    # Arrange
    gk = copy.deepcopy(_GEN_KWARGS)
    gk["max_agentic_steps"] = 5
    args = (
        "start",
        gk,
        _dummy_doc_to_visual,
        partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
        10,
        "agent",
        "dev",
    )
    inst = _make_instance("generate_until_agentic", args, doc_id=10, task="agent", split="dev")

    # Act
    ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split = inst.args

    # Assert
    assert ctx == "start"
    assert isinstance(gen_kwargs, dict)
    assert callable(doc_to_visual)
    assert callable(doc_to_text)
    assert doc_id == 10
    assert task == "agent"
    assert split == "dev"


# ---------------------------------------------------------------------------
# ConfigurableMessagesTask generate_until tests
# ---------------------------------------------------------------------------


def test_messages_task_generate_until_tuple_length_is_6():
    """ConfigurableMessagesTask generate_until produces 6-element tuple."""
    # Arrange
    args = ("msg_prompt", _dummy_doc_to_messages, copy.deepcopy(_GEN_KWARGS), 0, "mmmu_val", "test")
    inst = _make_instance("generate_until", args, doc_id=0, task="mmmu_val", split="test")

    # Act & Assert
    assert len(inst.args) == 6


def test_messages_task_generate_until_unpack_order():
    """ConfigurableMessagesTask generate_until unpacks in correct order."""
    # Arrange
    args = ("msg", _dummy_doc_to_messages, copy.deepcopy(_GEN_KWARGS), 5, "mmmu_val", "val")
    inst = _make_instance("generate_until", args, doc_id=5, task="mmmu_val", split="val")

    # Act
    ctx, doc_to_messages, gen_kwargs, doc_id, task, split = inst.args

    # Assert
    assert ctx == "msg"
    assert callable(doc_to_messages)
    assert isinstance(gen_kwargs, dict)
    assert doc_id == 5
    assert task == "mmmu_val"
    assert split == "val"


def test_messages_task_generate_until_doc_to_messages_is_callable():
    """ConfigurableMessagesTask generate_until has callable at index 1."""
    # Arrange
    args = ("msg_prompt", _dummy_doc_to_messages, copy.deepcopy(_GEN_KWARGS), 0, "mmmu_val", "test")
    inst = _make_instance("generate_until", args)

    # Act & Assert
    assert callable(inst.args[1])


def test_messages_task_generate_until_gen_kwargs_is_third_element():
    """ConfigurableMessagesTask generate_until has gen_kwargs at index 2, not 1."""
    # Arrange
    args = ("msg_prompt", _dummy_doc_to_messages, copy.deepcopy(_GEN_KWARGS), 0, "mmmu_val", "test")
    inst = _make_instance("generate_until", args)

    # Act & Assert
    assert isinstance(inst.args[2], dict)
    assert "temperature" in inst.args[2]


# ---------------------------------------------------------------------------
# ConfigurableMessagesTask agentic tests
# ---------------------------------------------------------------------------


def test_messages_task_agentic_tuple_length_is_7():
    """ConfigurableMessagesTask generate_until_agentic produces 7-element tuple."""
    # Arrange
    gk = copy.deepcopy(_GEN_KWARGS)
    gk["max_agentic_steps"] = 3
    args = (
        "msg_agentic",
        gk,
        _dummy_doc_to_visual,
        partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
        0,
        "agent_msg",
        "test",
    )
    inst = _make_instance("generate_until_agentic", args, doc_id=0, task="agent_msg", split="test")

    # Act & Assert
    assert len(inst.args) == 7


# ---------------------------------------------------------------------------
# Cross-type consistency checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_type, expected_len",
    [
        ("generate_until", 6),
        ("loglikelihood", 6),
        ("generate_until_multi_round", 7),
        ("generate_until_agentic", 7),
    ],
    ids=["gen_until", "loglikelihood", "multi_round", "agentic"],
)
def test_configurable_task_tuple_lengths(request_type, expected_len):
    """All ConfigurableTask output types produce the documented tuple length."""
    # Arrange
    if expected_len == 6 and request_type == "generate_until":
        args = ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, 0, "t", "test")
    elif expected_len == 6 and request_type == "loglikelihood":
        args = ("ctx", _dummy_doc_to_target, _dummy_doc_to_visual, 0, "t", "test")
    else:
        args = ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, _dummy_doc_to_text, 0, "t", "test")

    # Act
    inst = _make_instance(request_type, args)

    # Assert
    assert len(inst.args) == expected_len


def test_messages_vs_configurable_gen_kwargs_position():
    """ConfigurableTask has gen_kwargs at index 1; MessagesTask has it at index 2."""
    # Arrange - ConfigurableTask
    ct_args = ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, 0, "t", "test")
    ct = _make_instance("generate_until", ct_args)

    # Arrange - MessagesTask
    mt_args = ("ctx", _dummy_doc_to_messages, _GEN_KWARGS, 0, "t", "test")
    mt = _make_instance("generate_until", mt_args)

    # Act & Assert
    assert isinstance(ct.args[1], dict)
    assert callable(mt.args[1])
    assert isinstance(mt.args[2], dict)


def test_task_and_split_always_last_two():
    """For all output types, task_name and split are always the last two elements."""
    # Arrange
    test_cases = [
        ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, 0, "my_task", "my_split"),  # gen_until
        ("ctx", _dummy_doc_to_target, _dummy_doc_to_visual, 0, "my_task", "my_split"),  # loglik
        ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, _dummy_doc_to_text, 0, "my_task", "my_split"),  # agentic
    ]

    # Act & Assert
    for args in test_cases:
        inst = _make_instance("generate_until", args)
        assert inst.args[-1] == "my_split"
        assert inst.args[-2] == "my_task"

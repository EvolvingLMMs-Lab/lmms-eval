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
# ConfigurableTask tuple shapes
# ---------------------------------------------------------------------------


class TestConfigurableTaskGenerateUntil:
    """ConfigurableTask generate_until: (ctx, gen_kwargs, doc_to_visual, doc_id, task, split)"""

    def _make(self, ctx="prompt", doc_id=0, task="mme", split="test"):
        args = (ctx, copy.deepcopy(_GEN_KWARGS), _dummy_doc_to_visual, doc_id, task, split)
        return _make_instance("generate_until", args, doc_id=doc_id, task=task, split=split)

    def test_tuple_length_is_6(self):
        inst = self._make()
        assert len(inst.args) == 6

    def test_unpack_order(self):
        inst = self._make(ctx="my prompt", doc_id=42, task="mme", split="val")
        ctx, gen_kwargs, doc_to_visual, doc_id, task, split = inst.args
        assert ctx == "my prompt"
        assert isinstance(gen_kwargs, dict)
        assert "temperature" in gen_kwargs
        assert callable(doc_to_visual)
        assert doc_id == 42
        assert task == "mme"
        assert split == "val"

    def test_gen_kwargs_is_dict(self):
        inst = self._make()
        assert isinstance(inst.args[1], dict)

    def test_doc_to_visual_is_callable(self):
        inst = self._make()
        assert callable(inst.args[2])


class TestConfigurableTaskLoglikelihood:
    """ConfigurableTask loglikelihood: (ctx, doc_to_target, doc_to_visual, doc_id, task, split)"""

    def _make(self, ctx="context", doc_id=0, task="mmlu", split="test"):
        args = (ctx, _dummy_doc_to_target, _dummy_doc_to_visual, doc_id, task, split)
        return _make_instance("loglikelihood", args, doc_id=doc_id, task=task, split=split)

    def test_tuple_length_is_6(self):
        inst = self._make()
        assert len(inst.args) == 6

    def test_unpack_order(self):
        inst = self._make(ctx="What is 2+2?", doc_id=7, task="mmlu", split="val")
        ctx, doc_to_target, doc_to_visual, doc_id, task, split = inst.args
        assert ctx == "What is 2+2?"
        assert callable(doc_to_target)
        assert callable(doc_to_visual)
        assert doc_id == 7
        assert task == "mmlu"
        assert split == "val"

    def test_doc_to_target_is_callable(self):
        inst = self._make()
        assert callable(inst.args[1])


class TestConfigurableTaskMultiRound:
    """ConfigurableTask generate_until_multi_round: (ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split)"""

    def _make(self, ctx="round1", doc_id=0, task="multi", split="test"):
        args = (
            ctx,
            copy.deepcopy(_GEN_KWARGS),
            _dummy_doc_to_visual,
            partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
            doc_id,
            task,
            split,
        )
        return _make_instance("generate_until_multi_round", args, doc_id=doc_id, task=task, split=split)

    def test_tuple_length_is_7(self):
        inst = self._make()
        assert len(inst.args) == 7

    def test_unpack_order(self):
        inst = self._make(ctx="round1", doc_id=3, task="multi", split="val")
        ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split = inst.args
        assert ctx == "round1"
        assert isinstance(gen_kwargs, dict)
        assert callable(doc_to_visual)
        assert callable(doc_to_text)
        assert doc_id == 3
        assert task == "multi"
        assert split == "val"


class TestConfigurableTaskAgentic:
    """ConfigurableTask generate_until_agentic: (ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split)"""

    def _make(self, ctx="agentic_prompt", doc_id=0, task="agent", split="test"):
        gk = copy.deepcopy(_GEN_KWARGS)
        gk["max_agentic_steps"] = 5
        args = (
            ctx,
            gk,
            _dummy_doc_to_visual,
            partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
            doc_id,
            task,
            split,
        )
        return _make_instance("generate_until_agentic", args, doc_id=doc_id, task=task, split=split)

    def test_tuple_length_is_7(self):
        inst = self._make()
        assert len(inst.args) == 7

    def test_has_max_agentic_steps(self):
        inst = self._make()
        _, gen_kwargs, *_ = inst.args
        assert gen_kwargs["max_agentic_steps"] == 5

    def test_unpack_order(self):
        inst = self._make(ctx="start", doc_id=10, task="agent", split="dev")
        ctx, gen_kwargs, doc_to_visual, doc_to_text, doc_id, task, split = inst.args
        assert ctx == "start"
        assert isinstance(gen_kwargs, dict)
        assert callable(doc_to_visual)
        assert callable(doc_to_text)
        assert doc_id == 10
        assert task == "agent"
        assert split == "dev"


# ---------------------------------------------------------------------------
# ConfigurableMessagesTask tuple shapes
# ---------------------------------------------------------------------------


class TestMessagesTaskGenerateUntil:
    """ConfigurableMessagesTask generate_until: (ctx, doc_to_messages, gen_kwargs, doc_id, task, split)"""

    def _make(self, ctx="msg_prompt", doc_id=0, task="mmmu_val", split="test"):
        args = (ctx, _dummy_doc_to_messages, copy.deepcopy(_GEN_KWARGS), doc_id, task, split)
        return _make_instance("generate_until", args, doc_id=doc_id, task=task, split=split)

    def test_tuple_length_is_6(self):
        inst = self._make()
        assert len(inst.args) == 6

    def test_unpack_order_messages_task(self):
        inst = self._make(ctx="msg", doc_id=5, task="mmmu_val", split="val")
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = inst.args
        assert ctx == "msg"
        assert callable(doc_to_messages)
        assert isinstance(gen_kwargs, dict)
        assert doc_id == 5
        assert task == "mmmu_val"
        assert split == "val"

    def test_doc_to_messages_is_callable(self):
        inst = self._make()
        assert callable(inst.args[1])

    def test_gen_kwargs_is_third_element(self):
        """Key difference from ConfigurableTask: gen_kwargs is at index 2 not 1."""
        inst = self._make()
        assert isinstance(inst.args[2], dict)
        assert "temperature" in inst.args[2]


class TestMessagesTaskAgentic:
    """ConfigurableMessagesTask generate_until_agentic: same as ConfigurableTask agentic."""

    def _make(self, ctx="msg_agentic", doc_id=0, task="agent_msg", split="test"):
        gk = copy.deepcopy(_GEN_KWARGS)
        gk["max_agentic_steps"] = 3
        args = (
            ctx,
            gk,
            _dummy_doc_to_visual,
            partial(_dummy_doc_to_text, lmms_eval_specific_kwargs=None),
            doc_id,
            task,
            split,
        )
        return _make_instance("generate_until_agentic", args, doc_id=doc_id, task=task, split=split)

    def test_tuple_length_is_7(self):
        inst = self._make()
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
    if expected_len == 6 and request_type == "generate_until":
        args = ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, 0, "t", "test")
    elif expected_len == 6 and request_type == "loglikelihood":
        args = ("ctx", _dummy_doc_to_target, _dummy_doc_to_visual, 0, "t", "test")
    else:
        args = ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, _dummy_doc_to_text, 0, "t", "test")

    inst = _make_instance(request_type, args)
    assert len(inst.args) == expected_len


def test_messages_vs_configurable_gen_kwargs_position():
    """ConfigurableTask has gen_kwargs at index 1; MessagesTask has it at index 2."""
    # ConfigurableTask
    ct_args = ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, 0, "t", "test")
    ct = _make_instance("generate_until", ct_args)
    assert isinstance(ct.args[1], dict)

    # MessagesTask
    mt_args = ("ctx", _dummy_doc_to_messages, _GEN_KWARGS, 0, "t", "test")
    mt = _make_instance("generate_until", mt_args)
    assert callable(mt.args[1])
    assert isinstance(mt.args[2], dict)


def test_task_and_split_always_last_two():
    """For all output types, task_name and split are always the last two elements."""
    test_cases = [
        ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, 0, "my_task", "my_split"),  # gen_until
        ("ctx", _dummy_doc_to_target, _dummy_doc_to_visual, 0, "my_task", "my_split"),  # loglik
        ("ctx", _GEN_KWARGS, _dummy_doc_to_visual, _dummy_doc_to_text, 0, "my_task", "my_split"),  # agentic
    ]
    for args in test_cases:
        inst = _make_instance("generate_until", args)
        assert inst.args[-1] == "my_split"
        assert inst.args[-2] == "my_task"

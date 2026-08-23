"""Production request-construction contracts without dataset initialization."""

from types import SimpleNamespace

import pytest

from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask


def _doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return "prompt"


def _task(task_class, output_type):
    """Allocate a task shell so construct_requests never loads a dataset."""
    task = object.__new__(task_class)
    task.OUTPUT_TYPE = output_type
    task._config = SimpleNamespace(
        task="request-contract",
        generation_kwargs={"temperature": 0, "until": ["<stop>"]},
        doc_to_text=_doc_to_text,
    )
    task.lmms_eval_specific_kwargs = {"backend": "contract"}
    return task


def _request(task):
    return task.construct_requests(
        17,
        "context",
        metadata={"task": "request-contract", "doc_id": 17, "repeats": 1, "split": "validation"},
    )


def test_configurable_task_generate_until_constructs_simple_layout():
    task = _task(ConfigurableTask, "generate_until")

    request = _request(task)

    assert request.request_type == "generate_until"
    assert len(request.args) == 6
    assert request.args[0] == "context"
    assert isinstance(request.args[1], dict)
    assert request.args[1] == {"temperature": 0, "until": ["<stop>"]}
    assert request.args[1] is not task.config.generation_kwargs
    assert callable(request.args[2])
    assert request.args[3] == 17
    assert request.args[4] == "request-contract"
    assert request.args[5] == "validation"


def test_configurable_task_loglikelihood_constructs_simple_layout():
    task = _task(ConfigurableTask, "loglikelihood")

    request = _request(task)

    assert request.request_type == "loglikelihood"
    assert len(request.args) == 6
    assert request.args[0] == "context"
    assert callable(request.args[1])
    assert callable(request.args[2])
    assert request.args[3] == 17
    assert request.args[4] == "request-contract"
    assert request.args[5] == "validation"


@pytest.mark.parametrize("output_type", ["generate_until_multi_round", "generate_until_agentic"])
def test_configurable_task_constructs_multi_step_layout(output_type):
    task = _task(ConfigurableTask, output_type)

    request = _request(task)

    assert request.request_type == output_type
    assert len(request.args) == 7
    assert request.args[0] == "context"
    assert isinstance(request.args[1], dict)
    assert request.args[1] == {"temperature": 0, "until": ["<stop>"]}
    assert callable(request.args[2])
    assert callable(request.args[3])
    assert request.args[3].func is _doc_to_text
    assert request.args[3].keywords == {"lmms_eval_specific_kwargs": {"backend": "contract"}}
    assert request.args[4] == 17
    assert request.args[5] == "request-contract"
    assert request.args[6] == "validation"


@pytest.mark.parametrize("output_type", ["generate_until", "generate_until_multi_round"])
def test_messages_task_constructs_chat_generate_layout(output_type):
    task = _task(ConfigurableMessagesTask, output_type)

    request = _request(task)

    assert request.request_type == output_type
    assert len(request.args) == 6
    assert request.args[0] == "context"
    assert callable(request.args[1])
    assert isinstance(request.args[2], dict)
    assert request.args[2] == {"temperature": 0, "until": ["<stop>"]}
    assert request.args[2] is not task.config.generation_kwargs
    assert request.args[3] == 17
    assert request.args[4] == "request-contract"
    assert request.args[5] == "validation"


def test_messages_task_agentic_constructs_chat_multi_step_layout():
    task = _task(ConfigurableMessagesTask, "generate_until_agentic")

    request = _request(task)

    assert request.request_type == "generate_until_agentic"
    assert len(request.args) == 7
    assert request.args[0] == "context"
    assert isinstance(request.args[1], dict)
    assert request.args[1] == {"temperature": 0, "until": ["<stop>"]}
    assert callable(request.args[2])
    assert callable(request.args[3])
    assert request.args[3].func is _doc_to_text
    assert request.args[3].keywords == {"lmms_eval_specific_kwargs": {"backend": "contract"}}
    assert request.args[4] == 17
    assert request.args[5] == "request-contract"
    assert request.args[6] == "validation"

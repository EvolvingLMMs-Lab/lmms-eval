"""Opt-in scoring context must not change legacy callback invocation."""

from types import SimpleNamespace
from typing import Any

import pytest

from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask


@pytest.mark.parametrize("task_class", [ConfigurableTask, ConfigurableMessagesTask])
def test_explicit_settings_are_resolved_and_copied(task_class: type[ConfigurableTask]) -> None:
    def score(doc: dict, results: list[str], *, generation_kwargs: dict[str, Any], full_docs: list | None = None) -> dict:
        assert results == ["answer"]
        assert full_docs == ["full"]
        value = generation_kwargs["nested"]["value"]
        generation_kwargs["nested"]["value"] = 99
        return {"value": value}

    task = task_class.__new__(task_class)
    task.model_name = "qwen2_5_vl"
    task._config = SimpleNamespace(
        lmms_eval_specific_kwargs=None, model_specific_target_kwargs=None, model_specific_generation_kwargs={"qwen2_5_vl": {"nested": {"value": 7}}, "default": {"nested": {"value": 1}}}, generation_kwargs={}, process_results=score
    )
    task._prepare_model_specific_config()
    task.OUTPUT_TYPE = "generate_until"
    assert task.process_results({}, [" answer "], full_docs=["full"]) == {"value": 7}
    assert task.config.generation_kwargs["nested"]["value"] == 7
    task.model_name = "other"
    task._config.model_specific_generation_kwargs = {"default": {"nested": {"value": 1}}}
    task._prepare_model_specific_config()
    assert task.process_results({}, ["answer"], full_docs=["full"]) == {"value": 1}
    # CLI-style override is visible at scoring time, rather than captured at initialization.
    task.config.generation_kwargs = {"nested": {"value": 3}}
    assert task.process_results({}, ["answer"], full_docs=["full"]) == {"value": 3}


def test_legacy_and_kwargs_only_callbacks_are_unchanged() -> None:
    def legacy(doc: dict, results: list[str]) -> dict:
        return {"legacy": True}

    def kwargs_only(doc: dict, results: list[str], **kwargs: Any) -> dict:
        assert kwargs == {"full_docs": [1]}
        return {"legacy": True}

    task = ConfigurableTask.__new__(ConfigurableTask)
    task.OUTPUT_TYPE = "generate_until"
    task._config = SimpleNamespace(process_results=legacy, generation_kwargs={"setting": 1})
    assert task.process_results({}, ["answer"]) == {"legacy": True}
    task.config.process_results = kwargs_only
    assert task.process_results({}, ["answer"], full_docs=[1]) == {"legacy": True}


def test_callback_errors_are_not_reinterpreted_as_signature_errors() -> None:
    def score(doc: dict, results: list[str], *, generation_kwargs: dict[str, Any]) -> dict:
        raise TypeError("scoring error")

    task = ConfigurableTask.__new__(ConfigurableTask)
    task.OUTPUT_TYPE = "generate_until"
    task._config = SimpleNamespace(process_results=score, generation_kwargs={})
    # Bypass the existing retry decorator to assert the original exception directly.
    with pytest.raises(TypeError, match="scoring error"):
        ConfigurableTask.process_results.__wrapped__(task, {}, ["answer"])

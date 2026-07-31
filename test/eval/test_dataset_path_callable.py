"""`dataset_path: !function` — docs-from-code datasets for env-loop tasks."""

from __future__ import annotations

import datasets
import pytest

from lmms_eval.api.task import ConfigurableTask, TaskConfig


def _scenarios():
    return datasets.DatasetDict({"test": datasets.Dataset.from_list([{"env_id": "A", "seed": 1}, {"env_id": "B", "seed": 2}])})


def _bare_task(factory):
    task = ConfigurableTask.__new__(ConfigurableTask)
    task.DATASET_PATH = factory
    task._config = TaskConfig(test_split="test")
    return task


def test_download_builds_dataset_from_callable_dataset_path():
    task = _bare_task(_scenarios)

    task.download(None)

    assert list(task.dataset.keys()) == ["test"]
    assert task.dataset["test"].num_rows == 2
    assert task.dataset["test"][0]["env_id"] == "A"
    # the shared post-processing still runs for docs-from-code datasets
    assert task.dataset_no_image["test"].num_rows == 2


def test_download_rejects_factory_that_returns_non_datasetdict():
    task = _bare_task(lambda: [{"env_id": "A"}])

    # Call the undecorated method: the tenacity retry wrapper would otherwise
    # spin on the deterministic TypeError and wrap it in RetryError.
    with pytest.raises(TypeError, match="DatasetDict"):
        ConfigurableTask.download.__wrapped__(task, None)

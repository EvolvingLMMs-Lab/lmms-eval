"""Offline integration tests for local scoring of the public MMMU test split."""

from pathlib import Path

import datasets
import pytest

from lmms_eval.api.task import ConfigurableTask
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import load_yaml_config

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "mmmu"


@pytest.fixture
def local_task(monkeypatch):
    records = [
        {"id": "test_Accounting_1", "question": "Choose a value.", "question_type": "multiple-choice", "options": "['one', 'two']", "answer": "B"},
        {"id": "test_Math_1", "question": "Give the value.", "question_type": "open", "options": "[]", "answer": "0.5"},
        {"id": "test_Math_2", "question": "Give the shape.", "question_type": "open", "options": "[]", "answer": "hexagon"},
    ]
    captured = {}

    def download(task, dataset_kwargs):
        captured["path"] = task.DATASET_PATH
        captured["kwargs"] = dataset_kwargs
        task.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(records)})
        task.dataset_no_image = task.dataset

    monkeypatch.setattr(ConfigurableTask, "download", download)
    config = load_yaml_config(str(TASK_DIR / "mmmu_test_local.yaml"))
    return ConfigurableTask(config=config), captured


def test_local_test_scores_multiple_choice_and_open_answers(local_task):
    task, _ = local_task
    predictions = ["B", "The answer is 0.5.", "triangle"]
    metrics = [task.process_results(doc, [prediction])["mmmu_acc"] for doc, prediction in zip(task.eval_docs, predictions)]
    assert task.aggregation()["mmmu_acc"](metrics) == pytest.approx(0.66667)
    assert list(task.aggregation()) == ["mmmu_acc"]


def test_local_test_reads_public_labels_from_all_subjects(local_task):
    task, captured = local_task
    assert captured["path"] == "parquet"
    assert captured["kwargs"]["data_files"] == {"test": "hf://datasets/MMMU/MMMU/*/test-*.parquet"}
    assert task.config.test_split == "test"
    assert task.doc_to_target(task.eval_docs[0]) == "B"


def test_local_test_is_discoverable_and_legacy_submission_is_preserved():
    manager = TaskManager(verbosity="WARNING", include_defaults=False, include_path=str(TASK_DIR))
    assert "mmmu_test_local" in manager.all_tasks
    legacy = load_yaml_config(str(TASK_DIR / "mmmu_test.yaml"))
    local = load_yaml_config(str(TASK_DIR / "mmmu_test_local.yaml"))
    assert legacy["dataset_path"] == "lmms-lab-encoder/MMMU"
    assert [metric["metric"] for metric in legacy["metric_list"]] == ["submission"]
    assert local["generation_kwargs"] == legacy["generation_kwargs"]
    for hook in ("doc_to_text", "doc_to_visual", "process_results"):
        assert local[hook].__name__ == legacy[hook].__name__

"""TempCore: the temporally sensitive subset of six Video QA benchmarks.

Each ``tempcore_*`` task includes the original benchmark task and keeps only the
TempCore questions listed in ``tempcore_ids.json``. Prompts, video loading and
scoring are those of the original task.
"""

import json
from pathlib import Path
from typing import Callable

from datasets import Dataset

with open(Path(__file__).parent / "tempcore_ids.json") as f:
    TEMPCORE_IDS = {bench: set(ids) for bench, ids in json.load(f).items()}

MVBENCH_SUBTASKS = sorted({sample_id.split("/")[0] for sample_id in TEMPCORE_IDS["mvbench"]})


def _keep(dataset: Dataset, bench: str, key_fn: Callable[[dict], str]) -> Dataset:
    ids = TEMPCORE_IDS[bench]
    return dataset.filter(lambda doc: key_fn(doc) in ids)


def _mvbench_docs(subtask: str) -> Callable[[Dataset], Dataset]:
    """MVBench ids are ``<subtask>/<row index>`` in the upstream json order."""

    def process_docs(dataset: Dataset) -> Dataset:
        indices = [i for i in range(len(dataset)) if f"{subtask}/{i}" in TEMPCORE_IDS["mvbench"]]
        return dataset.select(indices)

    return process_docs


for _subtask in MVBENCH_SUBTASKS:
    globals()[f"mvbench_{_subtask}_docs"] = _mvbench_docs(_subtask)


def nextqa_docs(dataset: Dataset) -> Dataset:
    """Keep TempCore questions of ``nextqa_mc_test``."""
    return _keep(dataset, "nextqa", lambda doc: f"{doc['video']}/{doc['qid']}")


def egoschema_docs(dataset: Dataset) -> Dataset:
    """Keep TempCore questions of ``egoschema_subset``."""
    return _keep(dataset, "egoschema", lambda doc: doc["video_idx"])


def mlvu_docs(dataset: Dataset) -> Dataset:
    """Keep TempCore questions of ``mlvu_dev``."""
    return _keep(dataset, "mlvu", lambda doc: f"{doc['task_type']}/{doc['question_id']}")


def longvideobench_docs(dataset: Dataset) -> Dataset:
    """Keep TempCore questions of ``longvideobench_val_v``."""
    return _keep(dataset, "longvideobench", lambda doc: doc["id"])


def videomme_docs(dataset: Dataset) -> Dataset:
    """Keep TempCore questions of ``videomme``."""
    return _keep(dataset, "videomme", lambda doc: doc["question_id"])

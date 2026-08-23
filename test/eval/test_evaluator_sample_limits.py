import pytest

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.task import Task, TaskConfig
from lmms_eval.evaluator import evaluate
from lmms_eval.utils import create_iterator


class _NoOpAccelerator:
    def wait_for_everyone(self):
        pass


class _CountingLM(lmms):
    def __init__(self):
        super().__init__()
        self.accelerator = _NoOpAccelerator()

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        return ["answer"] * len(requests)

    def generate_until_multi_round(self, requests):
        raise NotImplementedError

    def clean(self):
        pass


class _CountingTask(Task):
    VERSION = "test"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, name, size):
        self._docs = [{"id": i} for i in range(size)]
        self.build_limits = []
        self.scored_doc_ids = []
        super().__init__()
        self._config = TaskConfig(task=name, test_split="test", num_fewshot=0, repeats=1, reasoning_tags=None)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        pass

    @property
    def task_name(self):
        return self.config.task

    @property
    def eval_docs(self):
        return self._docs

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self._docs

    def doc_iterator(self, *, rank=0, limit=None, world_size=1, offset=0):
        return create_iterator(
            enumerate(self._docs),
            rank=rank,
            limit=limit,
            world_size=world_size,
            offset=offset,
        )

    def doc_to_text(self, doc):
        return str(doc["id"])

    def doc_to_target(self, doc):
        return ""

    def build_all_requests(self, *, limit=None, offset=0, **kwargs):
        selected_docs = self._docs[offset:] if limit is None else self._docs[offset : offset + limit]
        self.build_limits.append(len(selected_docs))
        self._instances = []
        for doc in selected_docs:
            doc_id = doc["id"]
            instance = Instance(
                request_type="generate_until",
                arguments=(
                    "prompt",
                    {"until": []},
                    None,
                    doc_id,
                    self.task_name,
                    "test",
                ),
                idx=0,
                metadata={"task": self.task_name, "doc_id": doc_id, "repeats": 1},
            )
            instance.doc = doc
            self._instances.append(instance)

    def apply_filters(self):
        for instance in self._instances:
            instance.filtered_resps["none"] = instance.resps[0]

    def construct_requests(self, doc_id, ctx, **kwargs):
        raise NotImplementedError

    def process_results(self, doc, results):
        self.scored_doc_ids.append(doc["id"])
        return {"score": 1.0}

    def aggregation(self):
        return {"score": lambda scores: sum(scores) / len(scores)}

    def higher_is_better(self):
        return {"score": True}

    def dump_config(self):
        return {"num_fewshot": 0}


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(None, (3, 7)), (-1, (3, 7)), (2, (2, 2)), (0.5, (2, 4))],
)
def test_evaluate_resolves_limit_per_task_for_build_scoring_and_reporting(limit, expected):
    small = _CountingTask("small", size=3)
    large = _CountingTask("large", size=7)

    result = evaluate(
        _CountingLM(),
        {"small": small, "large": large},
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
    )

    if limit == 0.5:
        assert large.build_limits == [4]
        assert large.scored_doc_ids == [0, 1, 2, 3]
        assert result["n-samples"]["large"]["effective"] == 4
    assert (small.build_limits, large.build_limits) == ([expected[0]], [expected[1]])
    assert (small.scored_doc_ids, large.scored_doc_ids) == (
        list(range(expected[0])),
        list(range(expected[1])),
    )
    assert result["n-samples"] == {
        "small": {"original": 3, "effective": expected[0]},
        "large": {"original": 7, "effective": expected[1]},
    }

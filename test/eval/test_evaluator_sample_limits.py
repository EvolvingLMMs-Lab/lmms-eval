import pytest

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.task import ConfigurableTask, Task, TaskConfig
from lmms_eval.caching.response_cache import ResponseCache
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


class _LikelihoodTask(_CountingTask):
    OUTPUT_TYPE = "multiple_choice"

    def __init__(self) -> None:
        super().__init__("likelihood", size=2)
        self._config.output_type = self.OUTPUT_TYPE
        self._metric_fn_list = {"acc": None}
        self.multiple_input = False
        self.multiple_target = False

    def doc_to_choice(self, doc: dict) -> list[str]:
        return ["A", "B", "C", "D"]

    def doc_to_target(self, doc: dict) -> int:
        return doc["id"] + 1

    def build_all_requests(self, **kwargs: object) -> None:
        self._instances = []
        for doc in self._docs:
            for idx, choice in enumerate(self.doc_to_choice(doc)):
                instance = Instance(
                    request_type="loglikelihood",
                    arguments=(f"question {doc['id']}", choice, None, doc["id"], self.task_name, "test"),
                    idx=idx,
                    metadata={"task": self.task_name, "doc_id": doc["id"], "repeats": 1},
                )
                instance.doc = doc
                self._instances.append(instance)

    def process_results(self, doc: dict, results: list) -> dict:
        return ConfigurableTask.process_results(self, doc, results)

    def aggregation(self) -> dict:
        return {"acc": lambda scores: sum(scores) / len(scores)}

    def higher_is_better(self) -> dict:
        return {"acc": True}


class _LikelihoodLM(_CountingLM):
    def __init__(self) -> None:
        super().__init__()
        self.scored_requests = 0

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        self.scored_requests += len(requests)
        return [(0.25 if req.idx == req.doc_id + 1 else 2.5 + req.idx, req.idx == req.doc_id + 1) for req in requests]


@pytest.mark.parametrize("cached", [False, True])
def test_evaluate_preserves_likelihood_pairs_through_scoring_and_response_cache(tmp_path, cached: bool) -> None:
    model = _LikelihoodLM()
    cache = ResponseCache(str(tmp_path / "responses.db"), str(tmp_path / "audit.jsonl")) if cached else None
    try:
        # A warm SQLite cache returns JSON lists rather than the live tuples.
        for run in range(2 if cached else 1):
            result = evaluate(model, {"likelihood": _LikelihoodTask()}, bootstrap_iters=0, response_cache=cache, log_samples=True)
            assert result["results"]["likelihood"]["acc,none"] == 1.0
            assert model.scored_requests == 8
            samples = result["samples"]["likelihood"]
            assert len(samples) == 2
            for sample in samples:
                gold = sample["doc_id"] + 1
                expected = [[0.25 if idx == gold else 2.5 + idx, idx == gold] for idx in range(4)]
                assert [list(response) for response in sample["filtered_resps"]] == expected
                assert sample["token_counts"] == [None] * 4
                assert sample["acc"] == 1.0
                if cached and run == 1:
                    assert all(isinstance(response, list) for response in sample["filtered_resps"])
    finally:
        if cache is not None:
            cache.close()


@pytest.mark.parametrize(
    "output,expected_response,expected_counts",
    [
        ("answer", "answer", None),
        (GenerationResult("answer", TokenCounts(input_tokens=7, output_tokens=2)), "answer", {"input_tokens": 7, "output_tokens": 2}),
        (("answer", {"input_tokens": 7, "output_tokens": 2}), "answer", {"input_tokens": 7, "output_tokens": 2}),
        (["first round", "second round"], ["first round", "second round"], None),
    ],
)
def test_evaluate_retains_generation_output_normalization(output: object, expected_response: object, expected_counts: dict | None) -> None:
    class OutputLM(_CountingLM):
        def generate_until(self, requests: list[Instance]) -> list:
            return [output] * len(requests)

    result = evaluate(OutputLM(), {"generation": _CountingTask("generation", size=1)}, bootstrap_iters=0, log_samples=True)
    sample = result["samples"]["generation"][0]
    assert sample["filtered_resps"] == [expected_response]
    assert sample["token_counts"] == [expected_counts]

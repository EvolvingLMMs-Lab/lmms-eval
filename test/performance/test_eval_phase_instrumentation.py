from datasets import Dataset, DatasetDict

import lmms_eval._performance.recorder as recorder_module
import lmms_eval.evaluator as evaluator
from lmms_eval._performance.provenance import BaselineProvenance
from lmms_eval.api.instance import Instance
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.models.simple.dummy import Dummy
from lmms_eval.performance import BaselinePerformanceRecorder


class _InlineTask(ConfigurableTask):
    def __init__(self, task_name, size):
        self._documents = Dataset.from_list([{"question": f"q{index}", "answer": "A"} for index in range(size)])
        super().__init__(
            config={
                "task": task_name,
                "dataset_path": None,
                "test_split": "test",
                "output_type": "generate_until",
                "doc_to_text": "question",
                "doc_to_target": "answer",
                "doc_to_visual": lambda doc: [],
                "generation_kwargs": {"max_new_tokens": 1},
                "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
            }
        )

    def download(self, dataset_kwargs=None):
        self.dataset = DatasetDict({"test": self._documents})
        self.dataset_no_image = self.dataset


def _started_recorder(monkeypatch, tmp_path):
    fixed = BaselineProvenance("a" * 40, "sha256:" + "b" * 64, "sha256:" + "c" * 64, "fixture-hardware")
    monkeypatch.setattr(recorder_module, "capture_baseline_provenance", lambda root: fixed)
    recorder = BaselinePerformanceRecorder.capture(
        repo_root=tmp_path,
        legacy_arguments={"model": "dummy"},
        cache_state="disabled",
        repetition={"suite_id": "suite", "case_id": "case", "repetition_index": 0, "warmup": False},
    )
    recorder.start()
    return recorder


def test_evaluator_records_current_phase_boundaries_and_counts(monkeypatch, tmp_path):
    recorder = _started_recorder(monkeypatch, tmp_path)

    result = simple_evaluate(
        model="dummy",
        tasks=[_InlineTask("small", 4), _InlineTask("large", 10)],
        limit=2,
        bootstrap_iters=0,
        log_samples=True,
        performance_recorder=recorder,
    )
    recorder.finish()
    record = recorder.to_record()

    assert [phase["name"] for phase in record["phases"]] == [
        "model_load",
        "task_resolution",
        "request_build",
        "inference",
        "filter_and_normalize",
        "score",
        "aggregate",
        "reset",
    ]
    assert record["counters"] == {
        "failures": 0,
        "built_instances": 4,
        "selected_documents": 4,
        "inference_dispatches": 1,
        "responses": 4,
        "raw_outputs": 4,
        "normalized_outputs": 4,
        "scored_documents": 4,
    }
    assert "performance" not in result


def test_sparse_shard_request_summary_excludes_padding():
    real = Instance(
        request_type="generate_until",
        arguments=("prompt", {}, lambda doc: [], 17, "sparse", "test"),
        idx=0,
        metadata={"task": "sparse", "doc_id": 17, "repeats": 1},
    )
    padding = evaluator._clone_padding_request(real)

    assert evaluator._request_count_summary([padding]) == (0, 0, 1)
    assert evaluator._request_count_summary([real, padding]) == (1, 1, 1)


def test_evaluator_without_recorder_preserves_tuple_model_args():
    model_args = ("Return token: answer", ("compact", "full"))

    result = simple_evaluate(
        model=Dummy(),
        model_args=model_args,
        tasks=[_InlineTask("config-shape", 2)],
        limit=2,
        bootstrap_iters=0,
        log_samples=True,
    )

    assert result["config"]["model_args"] == model_args
    assert isinstance(result["config"]["model_args"], tuple)
    assert isinstance(result["config"]["model_args"][1], tuple)
    assert "performance" not in result

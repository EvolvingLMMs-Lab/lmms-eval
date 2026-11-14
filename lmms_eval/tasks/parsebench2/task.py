"""Native simple/chat generation with an annotation-free benchmark dataset."""

from typing import Any

from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask
from lmms_eval.tasks.parsebench2 import utils


class ParseBench2Task(ConfigurableMessagesTask):
    """Use standard generation/filter/logging contracts, preserving response bytes."""

    MODEL_NAME_AWARE = True

    def __init__(self, *args: Any, config: dict | None = None, model_name: str | None = None, task_type: str = "simple", **kwargs: Any) -> None:
        config = dict(config or {})
        config.pop("class", None)
        if task_type not in {"simple", "chat"}:
            raise ValueError("ParseBench-2 task_type must be simple or chat")
        self._request_protocol = task_type
        super().__init__(*args, config=config, model_name=model_name, **kwargs)
        if self.config.num_fewshot:
            raise ValueError("ParseBench-2 is a zero-shot task")

    def download(self, dataset_kwargs: dict | None = None) -> None:
        """The public HF JSONLs have incompatible annotation schemas across subsets."""
        import datasets

        utils.require_scorer()
        self.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(utils.load_inputs())})
        self.dataset_no_image = self.dataset

    def construct_requests(self, doc_id: int, ctx: Any, **kwargs: Any) -> Any:
        """Select the host model's supported request protocol."""
        if self._request_protocol == "simple":
            return ConfigurableTask.construct_requests(self, doc_id, ctx, **kwargs)
        return ConfigurableMessagesTask.construct_requests(self, doc_id, ctx, **kwargs)

    def process_results(self, doc: dict, results: list, full_docs: Any = None) -> dict:
        """Avoid ConfigurableTask's whitespace stripping for document transcription."""
        if results and isinstance(results[0], list):
            results = results[0]
        return utils.process_results(doc, results)

"""Keep the benchmark's plain Markdown boundary through framework scoring."""

from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask
from lmms_eval.tasks.olmocr_bench.utils import olmocr_bench_process_results


class OlmocrBenchTask(ConfigurableMessagesTask):
    """Preserve response whitespace used by first_n/last_n checks."""

    MODEL_NAME_AWARE = True

    def __init__(self, *args, config=None, model_name=None, task_type="simple", **kwargs):
        config = dict(config or {})
        config.pop("class", None)
        if task_type not in {"simple", "chat"}:
            raise ValueError("olmOCR-Bench task_type must be simple or chat")
        self._request_protocol = task_type
        super().__init__(*args, config=config, model_name=model_name, **kwargs)

    def construct_requests(self, doc_id, ctx, **kwargs):
        if self._request_protocol == "simple":
            return ConfigurableTask.construct_requests(self, doc_id, ctx, **kwargs)
        if self._request_protocol == "chat":
            return ConfigurableMessagesTask.construct_requests(self, doc_id, ctx, **kwargs)
        raise ValueError("olmOCR-Bench task_type must be simple or chat")

    def process_results(self, doc, results, full_docs=None):
        # ConfigurableTask strips each response before invoking the callback.
        # Leading/trailing whitespace is part of this benchmark's test window.
        if results and isinstance(results[0], list):
            results = results[0]
        return olmocr_bench_process_results(doc, ["" if result is None else result for result in results])

"""SwanLab logger for lmms-eval, a drop-in alternative to the wandb logger.

Mirrors the interface of ``lmms_eval.loggers.wandb_logger.WandbLogger`` so it can
be wired into the same evaluation lifecycle via ``--swanlab_args``:

    swanlab_logger = SwanLabLogger(**simple_parse_args_string(args.swanlab_args))
    swanlab_logger.post_init(results)
    swanlab_logger.log_eval_result()
    swanlab_logger.log_eval_samples(results["samples"])
    swanlab_logger.finish()

Authentication follows standard SwanLab mechanisms: set ``SWANLAB_API_KEY`` (and
optionally ``SWANLAB_HOST`` for a self-hosted instance); otherwise SwanLab's own
login flow / public cloud default applies. The default project is ``lmms-eval``,
overridable through the args string, e.g.
``--swanlab_args project=my-proj,exp_name=my-run,mode=cloud``.
"""

import copy
import json
import os
from typing import Any, Dict, List, Tuple

from lmms_eval.loggers.utils import _handle_non_serializable, remove_none_pattern

# SwanLab (0.7.x) has no Table/Artifact type, so per-task samples are logged as a
# capped swanlab.Text preview rather than a full browsable table.
_SAMPLE_PREVIEW_LIMIT = 10


class SwanLabLogger:
    def __init__(self, **kwargs) -> None:
        """Attaches to the active SwanLab run if one exists, otherwise passes kwargs to swanlab.init().

        Args:
            kwargs Optional[Any]: Arguments forwarded to ``swanlab.init``. ``exp_name``
                is accepted as an alias for SwanLab's ``experiment_name``. ``project``
                defaults to ``lmms-eval`` when not provided.

        Parse and log the results returned from evaluator.simple_evaluate() with:
            swanlab_logger.post_init(results)
            swanlab_logger.log_eval_result()
            swanlab_logger.log_eval_samples(results["samples"])
        """
        try:
            import swanlab
        except ImportError as e:
            raise ImportError("To use the SwanLab logging functionality please install swanlab.\n" "Run `pip install swanlab` (or `pip install lmms-eval[swanlab]`).") from e

        # `exp_name` is the user-facing knob (mirrors wandb's `name`); map it to
        # SwanLab's `experiment_name`.
        if "exp_name" in kwargs:
            kwargs["experiment_name"] = kwargs.pop("exp_name")
        kwargs.setdefault("project", "lmms-eval")
        self.swanlab_args: Dict[str, Any] = kwargs

        # Standard SwanLab auth: SWANLAB_API_KEY (+ optional SWANLAB_HOST for a
        # self-hosted instance). Without a key, SwanLab's own login flow applies.
        api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
        host = os.environ.get("SWANLAB_HOST", "").strip()
        if api_key:
            login_kwargs: Dict[str, Any] = {"api_key": api_key, "save": False}
            if host:
                login_kwargs["host"] = host
            swanlab.login(**login_kwargs)

        # Initialize (or attach to) a SwanLab run. swanlab.get_run() returns the
        # active run (or None) on <0.8, but raises RuntimeError on >=0.8 when no
        # run is active; treat "no active run" uniformly across both.
        try:
            active_run = swanlab.get_run()
        except Exception:
            active_run = None
        if active_run is None:
            self.run = swanlab.init(**self.swanlab_args)
        else:
            self.run = active_run

    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get("results", {}).keys())
        self.group_names: List[str] = list(results.get("groups", {}).keys())

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        self.task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": self.task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def _sanitize_results_dict(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Sanitize the results dictionary.

        Returns a tuple of (string-valued metrics, flattened numeric metrics keyed
        as ``{task}/{metric}``).
        """
        _results = copy.deepcopy(self.results.get("results", dict()))

        # Remove None from the metric string name
        tmp_results = copy.deepcopy(_results)
        for task_name in self.task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)

        # remove string valued keys from the results dict
        string_summary = {}
        for task in self.task_names:
            task_result = _results.get(task, dict())
            for metric_name, metric_value in task_result.items():
                if isinstance(metric_value, str):
                    string_summary[f"{task}/{metric_name}"] = metric_value

        for summary_metric, summary_value in string_summary.items():
            _task, _summary_metric = summary_metric.split("/")
            _results[_task].pop(_summary_metric)

        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f"{task_name}/{metric_name}"] = metric_value
                _results[task_name].pop(metric_name)
        for task in self.task_names:
            _results.pop(task)

        return string_summary, _results

    def log_eval_result(self) -> None:
        """Log evaluation results to SwanLab."""
        # Log configs to the run config.
        configs = self._get_config()
        self.run.config.update(configs)

        string_summary, swanlab_results = self._sanitize_results_dict()
        # SwanLab has no run.summary; keep string-valued metrics in the run config
        # for provenance instead.
        if string_summary:
            self.run.config.update(string_summary)
        # Log the numeric evaluation metrics as SwanLab scalars.
        if swanlab_results:
            self.run.log(swanlab_results)
        # NOTE: wandb's results-as-Table and results-as-JSON-artifact steps have no
        # SwanLab (0.7.x) equivalent; the scalar metrics above cover the leaderboard
        # use-case.

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log evaluation samples to SwanLab.

        SwanLab (0.7.x) has no Table/Artifact type, so each (ungrouped) task's
        samples are logged as a sample-count scalar plus a capped JSON preview
        rendered through ``swanlab.Text``.

        Args:
            samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
        """
        import swanlab

        task_names: List[str] = [x for x in self.task_names if x not in self.group_names]
        for task_name in task_names:
            eval_preds = samples.get(task_name)
            if not eval_preds:
                continue

            self.run.log({f"eval_samples/{task_name}/num_samples": len(eval_preds)})

            preview = json.dumps(
                eval_preds[:_SAMPLE_PREVIEW_LIMIT],
                indent=2,
                default=_handle_non_serializable,
                ensure_ascii=False,
            )
            self.run.log({f"eval_samples/{task_name}/preview": swanlab.Text(f"```json\n{preview}\n```")})

    def finish(self) -> None:
        """Finish the SwanLab run."""
        import swanlab

        swanlab.finish()

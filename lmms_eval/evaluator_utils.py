# credit to https://github.com/EleutherAI/lm-evaluation-harness
import collections
import importlib
import inspect
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from lmms_eval.api.group import ConfigurableGroup
from lmms_eval.api.metrics import (
    aggregate_subtask_metrics,
    clustered_stderr,
    consensus_accuracy,
    consistency_rate,
    expected_accuracy,
    internal_variance,
    paired_ttest,
    pooled_sample_stderr,
    stderr_for_metric,
)
from lmms_eval.api.task import Task
from lmms_eval.utils import eval_logger, positional_deprecated


class TaskOutput:
    """
    Wrapper class for Task outputs.It contains various attributes and methods to manage and calculate metrics for the task.

        Attributes:
            task (object): The task object.
            task_name (str): The name of the task.
            task_config (dict): The configuration of the task.
            version (str): The version of the task.
            group_name (str): The name of the task group.
            n_shot (int): The number of shots for the task.
            task_alias (str): The alias of the task.
            group_alias (str): The alias of the task group.
            is_group (bool): Indicates if the task is a group.
            logged_samples (list): The list of logged samples.
            sample_len (int): The number of documents scored for the task.
            sample_counts (dict): The number of items behind each "metric,filter" key; a metric can cover fewer documents than ``sample_len``.
            sample_metrics (defaultdict): The dictionary of samples' metrics.
            agg_metrics (defaultdict): The dictionary of aggregate metrics.

        Methods:
            from_taskdict(cls, task_name: str, task):
                Creates a TaskOutput instance from a task dictionary.

            calculate_aggregate_metric(bootstrap_iters=100000) -> None:
                Calculates the aggregate metrics for the task.
    """

    def __init__(
        self,
        task=None,
        task_name=None,
        task_config=None,
        version=None,
        group_name=None,
        n_shot=None,
        task_alias=None,
        group_alias=None,
        is_group=None,
    ):
        self.task = task
        self.task_config = task_config
        self.task_name = task_name
        self.group_name = group_name
        self.version = version
        self.n_shot = n_shot
        self.task_alias = task_alias
        self.group_alias = group_alias
        self.is_group = is_group
        self.logged_samples = []
        self.sample_len = None
        self.sample_counts = {}
        self.sample_metrics = collections.defaultdict(list)
        self.per_sample_metrics = collections.defaultdict(list)  # Per-sample scores for stability metrics
        self.agg_metrics = collections.defaultdict(list)
        self.args = None

    @classmethod
    def from_taskdict(cls, task_name: str, task):
        if isinstance(task, tuple):
            group_name, task = task
        else:
            group_name = None
        if not task:
            # these gets filtered out in get_task_list
            # once they are added to group hierarchy
            is_group = True
            return cls(task=task, task_name=task_name, is_group=is_group, group_name=group_name)
        version = task.VERSION
        task_config = dict(task.dump_config())
        if (n_shot := task_config.get("num_fewshot")) == 0:
            meta_config = task_config.get("metadata", {})
            if isinstance(meta_config, dict):
                n_shot = meta_config.get("num_fewshot", 0)
            else:
                eval_logger.info(f"No metadata found in task config for {task_name}, using default n_shot=0")
                n_shot = 0
        task_alias = task_config.get("alias")
        group_alias = task_config.get("group_alias")
        return cls(
            task=task,
            task_name=task_name,
            task_config=task_config,
            group_name=group_name,
            version=version,
            n_shot=n_shot,
            task_alias=task_alias,
            group_alias=group_alias,
        )

    def calculate_aggregate_metric(self, bootstrap_iters=100000) -> None:
        for (metric, filter_key), items in self.sample_metrics.items():
            if metric in self.task.aggregation():
                agg_fn = self.task.aggregation()[metric]
                metric_key = f"{metric},{filter_key}"
                if "args" in inspect.signature(agg_fn).parameters:
                    self.agg_metrics[metric_key] = agg_fn(items, args=self.task.args)
                else:
                    self.agg_metrics[metric_key] = agg_fn(items)
                self.sample_counts[metric_key] = len(items)
                if isinstance(bootstrap_iters, int):
                    stderr_fn = stderr_for_metric(
                        metric=agg_fn,
                        bootstrap_iters=min(bootstrap_iters, 100) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters,
                    )
                    self.agg_metrics[f"{metric}_stderr,{filter_key}"] = stderr_fn(items) if (stderr_fn and len(items) > 1) else "N/A"
                else:
                    raise ValueError(f"Received bootstrap_iters '{bootstrap_iters}' but expected an integer. Set to 0 to turn off stderr calculations.")

    def calculate_clt_aggregate_metric(self) -> None:
        """Calculate CLT-based standard errors (naive and clustered)."""
        # Get cluster_key from task config (e.g., "videoID" for videomme)
        cluster_key = self.task_config.get("cluster_key") if self.task_config else None
        score_key = self.task_config.get("score_key", "score") if self.task_config else "score"

        for (metric, filter_key), items in self.sample_metrics.items():
            if metric not in self.task.aggregation():
                continue
            # Extract scores and cluster_ids from items
            # Convention: dict items should have the score_key field (0/1) and optionally the field specified by cluster_key
            numeric_items = []
            cluster_ids = []
            for x in items:
                if isinstance(x, (int, float)):
                    numeric_items.append(x)
                    cluster_ids.append(None)
                elif isinstance(x, dict) and score_key in x:
                    numeric_items.append(x[score_key])
                    cluster_ids.append(x.get(cluster_key) if cluster_key else None)
            n = len(numeric_items)
            # Naive CLT stderr: std / sqrt(n)
            self.agg_metrics[f"{metric}_stderr_clt,{filter_key}"] = np.std(numeric_items, ddof=1) / np.sqrt(n) if n > 1 else "N/A"
            # Clustered stderr: only if cluster_ids are available and have >1 unique clusters
            valid_clusters = [c for c in cluster_ids if c is not None]
            if valid_clusters and len(set(valid_clusters)) > 1 and n > 1:
                self.agg_metrics[f"{metric}_stderr_clustered,{filter_key}"] = clustered_stderr(numeric_items, cluster_ids)
            else:
                self.agg_metrics[f"{metric}_stderr_clustered,{filter_key}"] = "N/A"

    def calculate_stability_metrics(self) -> None:
        """Calculate model stability metrics (EA, CA, IV, CR) when repeats > 1.

        These metrics measure model consistency across multiple samples per question.
        Only computed when repeats > 1 (k-samples mode).

        Uses per_sample_metrics which contains per-sample scores grouped by doc_id.
        Each entry in per_sample_metrics is a list of k scores for one question.
        """
        repeats = self.task_config.get("repeats", 1) if self.task_config else 1
        if repeats <= 1:
            return  # Skip if not in k-samples mode

        score_key = self.task_config.get("score_key", "score") if self.task_config else "score"

        for (metric, filter_key), items in self.per_sample_metrics.items():
            if metric not in self.task.aggregation():
                continue

            # items is already grouped by doc_id, each element is a list of k scores
            scores_per_question = []
            for sample_scores in items:
                if not isinstance(sample_scores, list):
                    # Fallback: if not a list, skip with warning
                    eval_logger.warning(f"Stability metrics: expected list of scores per question, got {type(sample_scores)}. Skipping.")
                    continue

                question_scores = []
                for x in sample_scores:
                    if isinstance(x, (int, float)):
                        question_scores.append(float(x))
                    elif isinstance(x, dict) and score_key in x:
                        question_scores.append(float(x[score_key]))
                    else:
                        eval_logger.debug(f"Stability metrics: cannot extract score from {type(x)}: {x}")

                if question_scores:
                    scores_per_question.append(question_scores)

            if not scores_per_question:
                eval_logger.warning(f"Stability metrics: no valid scores found for metric {metric}. Skipping.")
                continue

            # Calculate stability metrics
            self.agg_metrics[f"{metric}_expected_accuracy,{filter_key}"] = expected_accuracy(scores_per_question)
            self.agg_metrics[f"{metric}_consensus_accuracy,{filter_key}"] = consensus_accuracy(scores_per_question)
            self.agg_metrics[f"{metric}_internal_variance,{filter_key}"] = internal_variance(scores_per_question)
            self.agg_metrics[f"{metric}_consistency_rate,{filter_key}"] = consistency_rate(scores_per_question)

    def __repr__(self):
        return f"TaskOutput(task_name={self.task_name}, group_name={self.group_name}, version={self.version}, n_shot={self.n_shot}, task_alias={self.task_alias}, group_alias={self.group_alias})"


def get_task_list(task_dict: dict) -> List[TaskOutput]:
    outputs = []
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            _outputs = get_task_list(task_obj)
            outputs.extend(_outputs)
        else:
            task_output = TaskOutput.from_taskdict(task_name, task_obj)
            outputs.append(task_output)

    return outputs


def get_subtask_list(task_dict, task_root=None, depth=0):
    subtask_list = {}
    for group_obj, task_obj in task_dict.items():
        if isinstance(group_obj, ConfigurableGroup):
            # group_name = group_obj.group_name
            group_name = group_obj.group_name
        else:
            group_name = group_obj
        if isinstance(task_obj, dict):
            _subtask_list = get_subtask_list(task_obj, task_root=group_name, depth=depth + 1)
            if task_root:
                subtask_list.setdefault((task_root, depth), []).extend([_task for (_task, _depth) in _subtask_list.keys() if (_depth - 1) == depth])

            subtask_list = {**subtask_list, **_subtask_list}
        else:
            if isinstance(task_obj, ConfigurableGroup):
                # group_or_task_name = task_obj.group_name
                group_or_task_name = task_obj.group_name
            elif isinstance(task_obj, Task):
                # group_or_task_name = task_obj.task_name
                group_or_task_name = task_obj.task_name

            if task_root is None:
                subtask_list.setdefault((group_or_task_name, depth), [])
            else:
                subtask_list.setdefault((task_root, depth), []).append(group_or_task_name)

    if depth == 0:
        _subtask_list = {}
        for group_key, task_list in subtask_list.items():
            group_name, depth = group_key
            _subtask_list[group_name] = task_list
        subtask_list = _subtask_list

    return subtask_list


def print_writeout(task) -> None:
    """Print first few documents for debugging purposes.

    DEPRECATED: This function is deprecated and will be removed in v0.5.0.
    Use log_samples functionality instead for better debugging capabilities.

    WARNING: This function only prints the first few documents to console
    and can significantly impact performance during evaluations.
    """
    for inst in task.instances:
        # print the prompt for the first few documents
        if inst.doc_id < 1:
            # Handle cases where inst.doc might be None (e.g., when using log_samples)
            target = "N/A (document is None)" if inst.doc is None else task.doc_to_target(inst.doc)
            eval_logger.info(
                f"Task: {task}; document {inst.doc_id}; context prompt (starting on next line):\
    \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{target}\n(end of target on previous line)"
            )
            eval_logger.info(f"Request: {str(inst)}")


def get_sample_size(task, limit: Optional[Union[int, float]]) -> Union[int, None]:
    if limit is None or limit == -1:
        return None
    if limit < 0:
        raise ValueError(f"limit must be -1 or non-negative, got {limit}")
    if 0 < limit < 1.0:
        return int(math.ceil(len(task.eval_docs) * limit))
    return int(limit)


def prepare_print_tasks(
    task_dict: dict,
    results: dict,
    task_depth=0,
    group_depth=0,
) -> Tuple[dict, dict]:
    """
    @param task_dict: Dictionary representing the group hierarchy of tasks. Each key is a group name and its
    value is a list of task names.
    @param results: Dictionary containing the results of each task. Each key is a
    group name and its value is a dictionary of task results.
    @param task_depth: The indentation level for printing the task
    hierarchy. Default is 0.
    @param group_depth: The indentation level for printing the group
    hierarchy. Default is 0.
    @return: A tuple of two dictionaries: results_agg and groups_agg. results_agg contains
    aggregated results for each task, and groups_agg contains aggregated results for each group.

    Prepares the task hierarchy and aggregates the results for each task and group recursively for printing.
    """

    def _sort_task_dict(task_dict):
        """
        Helper utility. Sorts the task dict at the current level of the hierarchy based on alphabetized task name.
        Required so that we end up sorting within each sub-header correctly.
        """

        return dict(
            sorted(
                task_dict.items(),
                key=lambda item: item[0].group_name if isinstance(item[0], ConfigurableGroup) else item[0],
            )
        )

    task_agg = collections.defaultdict(dict)
    group_agg = collections.defaultdict(dict)
    task_dict = _sort_task_dict(task_dict)
    for task_or_group_name, task_or_group_obj in task_dict.items():
        tab_string = " " * task_depth + "- " if task_depth > 0 else ""
        if isinstance(task_or_group_name, ConfigurableGroup):
            # string_name = task_or_group_name.group_name
            name = task_or_group_name.group_name
            from_configurable_group = True
            task_or_group_obj = _sort_task_dict(task_or_group_obj)
        elif isinstance(task_or_group_name, str):
            name = task_or_group_name
            if isinstance(task_or_group_obj, Task):
                # string_name = task_or_group_obj.task_name
                name = task_or_group_obj.task_name
            from_configurable_group = False

        task_agg[name] = results[name].copy()
        if from_configurable_group:
            if task_or_group_name.group_alias is not None:
                alias = task_or_group_name.group_alias
            else:
                alias = task_or_group_name.group
        else:
            if "alias" in task_agg[name]:
                alias = task_agg[name]["alias"]
            else:
                alias = name

        task_agg[name]["alias"] = tab_string + alias
        if "samples" in task_agg[name]:
            task_agg[name].pop("samples")
        task_agg[name].pop("sample_counts", None)

        if from_configurable_group and (" " not in results[name]):
            group_tab_string = " " * group_depth + "- " if group_depth > 0 else ""
            group_agg[name] = results[name].copy()
            group_agg[name]["alias"] = group_tab_string + alias
            if "samples" in group_agg[name]:
                group_agg[name].pop("samples")
            group_agg[name].pop("sample_counts", None)

        if isinstance(task_or_group_obj, dict):
            task_depth += 1
            group_depth += 1
            _task_agg, _group_agg = prepare_print_tasks(task_or_group_obj, results, task_depth, group_depth)
            task_agg = {
                **task_agg,
                **_task_agg,
            }
            group_agg = {**group_agg, **_group_agg}
            task_depth -= 1
            group_depth -= 1
    return task_agg, group_agg


def consolidate_results(
    eval_tasks: List[TaskOutput],
) -> Tuple[dict, dict, dict, dict, dict, dict]:
    """
    @param eval_tasks: list(TaskOutput).
    @return: A tuple containing the consolidated results, samples, configs, versions, and num_fewshot.

    Consolidates the results of multiple evaluation tasks into a single structure.

    The method iterates over each evaluation instance and extracts relevant information to create the consolidated
    results structure. The consolidated results structure has the following properties:

    - results: A defaultdict with task names as keys and dictionaries as values. Each dictionary contains
    metric/filter pairs as keys and corresponding metric values as values. The "alias" key is used to store task
    aliases specified in the task configuration.
    - samples: A defaultdict with task names as keys and lists of log samples as values.
    - configs: A defaultdict with task names as keys and task configurations as values.
    - versions: A defaultdict with task names as keys and task versions as values.
    - num_fewshot: A defaultdict with task names as keys and number of few-shot samples as values.
    - higher_is_better: A defaultdict with task names as keys and indicators of whether higher values are better
    for each metric as values.

    The method then returns the consolidated results, samples, configs, versions, and num_fewshot as a tuple.
    """
    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    # Tracks the YAML configs of all chosen task
    configs = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Track `higher_is_better` for each metric
    higher_is_better = collections.defaultdict(dict)

    for task_output in eval_tasks:
        if "task_alias" in (task_config := task_output.task_config):
            results[task_output.task_name]["alias"] = task_config["task_alias"]
        else:
            results[task_output.task_name]["alias"] = task_output.task_name
        if group_alias := task_output.group_alias:
            if group_alias not in results and (group_name := task_output.group_name):
                results[group_name]["alias"] = group_alias
        num_fewshot[task_output.task_name] = task_output.n_shot
        configs[task_output.task_name] = task_output.task_config
        versions[task_output.task_name] = task_output.version
        samples[task_output.task_name] = task_output.logged_samples
        higher_is_better[task_output.task_name] = task_output.task.higher_is_better()
        for (metric, filter_key), items in task_output.sample_metrics.items():
            metric_key = f"{metric},{filter_key}"
            results[task_output.task_name][metric_key] = task_output.agg_metrics[metric_key]
            results[task_output.task_name]["samples"] = task_output.sample_len
            results[task_output.task_name].setdefault("sample_counts", {})[metric_key] = task_output.sample_counts.get(metric_key, task_output.sample_len)
            results[task_output.task_name][f"{metric}_stderr,{filter_key}"] = task_output.agg_metrics[f"{metric}_stderr,{filter_key}"]
            # Output CLT stderr
            clt_key = f"{metric}_stderr_clt,{filter_key}"
            if clt_key in task_output.agg_metrics:
                results[task_output.task_name][clt_key] = task_output.agg_metrics[clt_key]
            # Output clustered stderr
            clustered_key = f"{metric}_stderr_clustered,{filter_key}"
            if clustered_key in task_output.agg_metrics:
                results[task_output.task_name][clustered_key] = task_output.agg_metrics[clustered_key]
            # Output stability metrics (EA, CA, IV, CR) when in k-samples mode
            for stat_suffix in [
                "expected_accuracy",
                "consensus_accuracy",
                "internal_variance",
                "consistency_rate",
            ]:
                stat_key = f"{metric}_{stat_suffix},{filter_key}"
                if stat_key in task_output.agg_metrics:
                    results[task_output.task_name][stat_key] = task_output.agg_metrics[stat_key]
    return results, samples, configs, versions, num_fewshot, higher_is_better


def _resolve_group_postprocess_callable(raw):
    """Resolve group postprocess_results value to a callable.

    Supports a callable directly (from YAML !function) or a string import path
    such as "lmms_eval.tasks.sitebench.utils.sitebench_merge_results".
    """
    if raw is None:
        return None
    if callable(raw):
        return raw
    if isinstance(raw, str):
        try:
            module_name, func_name = raw.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name)
            if callable(fn):
                return fn
            eval_logger.warning(f"Group postprocess_results '{raw}' resolved to non-callable {fn}")
            return None
        except Exception as e:
            eval_logger.warning(f"Failed to resolve group postprocess_results '{raw}': {e}")
            return None
    eval_logger.warning(f"Unsupported postprocess_results type {type(raw)}: {raw}")
    return None


def _group_postprocess_context(
    *,
    group_name,
    subtask_names,
    results,
    samples=None,
    configs=None,
    versions=None,
    output_dir=None,
    model_output_dir=None,
    sample_files=None,
    group_config=None,
    group_metadata=None,
):
    return {
        "group_name": group_name,
        "group": group_name,
        "subtask_names": subtask_names,
        "subtasks": subtask_names,
        "task_names": subtask_names,
        "results": results,
        "samples": samples,
        "configs": configs,
        "versions": versions,
        "output_dir": output_dir,
        "output_path": output_dir,
        "model_output_dir": model_output_dir,
        "model_result_dir": model_output_dir,
        "sample_files": sample_files,
        "sample_paths": sample_files,
        "group_config": group_config,
        "config": group_config,
        "group_metadata": group_metadata,
        "metadata": group_metadata,
    }


def _bind_group_postprocess_call(hook, context):
    """Return a zero-arg caller whose arguments already match ``hook``.

    Binding happens before the hook runs. A TypeError raised inside the hook
    must not be treated as a bad signature and must not cause a second call.
    """
    group_name = context["group_name"]
    subtask_names = context["subtask_names"]
    results = context["results"]
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):

        def _call_without_signature():
            return hook(group_name, subtask_names, results)

        return _call_without_signature

    parameters = list(signature.parameters.values())
    has_var_keyword = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if has_var_keyword:
        filtered = dict(context)
    else:
        filtered = {key: value for key, value in context.items() if key in signature.parameters}

    try:
        signature.bind(**filtered)
    except TypeError:
        positional = [parameter for parameter in parameters if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        if len(positional) >= 3:
            positional_names = [parameter.name for parameter in positional[:3]]
            rest = {key: value for key, value in filtered.items() if key not in positional_names}
            try:
                signature.bind(group_name, subtask_names, results, **rest)
            except TypeError:
                rest = None
            else:

                def _call_positional_with_rest(rest=rest):
                    return hook(group_name, subtask_names, results, **rest)

                return _call_positional_with_rest
        try:
            signature.bind(group_name, subtask_names, results)
        except TypeError as exc:
            eval_logger.warning(f"Group postprocess hook '{group_name}' arguments do not match its signature: {exc}")
            return None

        def _call_positional():
            return hook(group_name, subtask_names, results)

        return _call_positional

    def _call_keywords():
        return hook(**filtered)

    return _call_keywords


def _invoke_group_postprocess(
    hook,
    *,
    group_name,
    subtask_names,
    results,
    samples=None,
    configs=None,
    versions=None,
    output_dir=None,
    model_output_dir=None,
    sample_files=None,
    group_config=None,
    group_metadata=None,
):
    """Invoke a group postprocess hook once with the arguments it accepts."""
    if hook is None:
        return None
    context = _group_postprocess_context(
        group_name=group_name,
        subtask_names=subtask_names,
        results=results,
        samples=samples,
        configs=configs,
        versions=versions,
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        sample_files=sample_files,
        group_config=group_config,
        group_metadata=group_metadata,
    )
    call = _bind_group_postprocess_call(hook, context)
    if call is None:
        return None
    try:
        return call()
    except Exception as exc:
        eval_logger.warning(f"Group postprocess hook '{group_name}' failed: {exc}")
        return None


def _apply_group_postprocess(
    group_obj,
    group_name,
    results,
    versions,
    show_group_table,
    task_aggregation_list,
    *,
    samples,
    configs,
    output_dir,
    model_output_dir,
    sample_files,
    aggregate_metric_list,
):
    """Merge an opt-in group hook into ``results``.

    Returns ``show_group_table`` unchanged when the group has no hook. The
    caller must already have finished the default aggregation path.
    """
    if group_obj is None:
        return show_group_table
    raw_hook = getattr(group_obj._config, "postprocess_results", None)
    if not raw_hook:
        return show_group_table

    hook_fn = _resolve_group_postprocess_callable(raw_hook)
    if hook_fn is None:
        return show_group_table

    task_list = list(task_aggregation_list.get(group_name, []))
    group_config = group_obj._config.to_dict(keep_callable=True)
    group_metadata = group_config.get("metadata")
    if "alias" not in results[group_name]:
        results[group_name]["alias"] = group_name
    hook_result = _invoke_group_postprocess(
        hook_fn,
        group_name=group_name,
        subtask_names=task_list,
        results=results,
        samples=samples,
        configs=configs,
        versions=versions,
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        sample_files=sample_files,
        group_config=group_config,
        group_metadata=group_metadata,
    )
    if isinstance(hook_result, dict) and hook_result:
        for key, value in hook_result.items():
            results[group_name][key] = value
        results[group_name].pop(" ", None)
        show_group_table = True
        if aggregate_metric_list is None and "samples" not in results[group_name] and task_list:
            sizes = [results[task].get("samples", 0) for task in task_list if "samples" in results[task]]
            if sizes:
                results[group_name]["samples"] = sum(sizes)
        if aggregate_metric_list is None and group_metadata is not None and group_name not in versions:
            versions[group_name] = group_metadata.get("version", None)
    elif hook_result is not None and not isinstance(hook_result, dict):
        eval_logger.warning(f"Group postprocess hook '{group_name}' returned non-dict {type(hook_result)}, ignoring")
    return show_group_table


def consolidate_group_results(
    results,
    versions,
    task_dict,
    task_root=None,
    show_group_table=False,
    task_aggregation_list=None,
    samples=None,
    configs=None,
    output_dir=None,
    model_output_dir=None,
    sample_files=None,
) -> Tuple[dict, dict, bool, Union[None, dict]]:
    """
    (Recursively) calculates groups' aggregated metrics and updates the results and versions dictionaries with this info.

    @return: a tuple [results, versions, show_group_table, task_aggregation_list] with formats described below:

    - results: A defaultdict with task names (and, after this function is called, group names of
    groups that perform aggregation) as keys, and dictionaries with "alias" and metric,filter_name pairs as keys.
    - versions: A defaultdict with task names (and, after this function is called, group names of
    groups that perform aggregation) as keys, and float values representing the task or group's version if a version is specified. (defaulting to None).
    - show_group_table: a boolean which is true if there exists a group that requires printing of its aggregated scores in a group table.
    - task_aggregation_list: a defaultdict listing the subtasks to average over to produce a given group's end metric.

    The method then returns the updated results, versions, show_group_table, and task_aggregation_list as a tuple.
    In the top-level invocation of this function, task_aggregation_list is ignored.
    """
    if task_root is None:
        task_root = {}

    if task_aggregation_list is None:
        task_aggregation_list = {}

    for group_or_task, group_or_task_info in task_dict.items():
        # Convert to string
        group_obj = group_or_task if isinstance(group_or_task, ConfigurableGroup) else None
        if group_obj is not None:
            group_config = group_obj.config
            group_or_task = group_obj.group_name
        else:
            group_config = None

        if isinstance(group_or_task_info, Task):
            if task_root:
                task_aggregation_list.setdefault(task_root, []).append(group_or_task_info.task_name)
        else:
            (
                results,
                versions,
                show_group_table,
                _task_aggregation_list,
            ) = consolidate_group_results(
                results,
                versions,
                group_or_task_info,
                group_or_task,
                show_group_table,
                task_aggregation_list,
                samples=samples,
                configs=configs,
                output_dir=output_dir,
                model_output_dir=model_output_dir,
                sample_files=sample_files,
            )
            if task_root:
                task_aggregation_list.setdefault(task_root, []).extend(task_aggregation_list.get(group_or_task, []))

            hook_kwargs = {
                "samples": samples,
                "configs": configs,
                "output_dir": output_dir,
                "model_output_dir": model_output_dir,
                "sample_files": sample_files,
            }
            aggregate_metric_list = None if group_config is None else group_config["aggregate_metric_list"]
            if (group_config is None) or (aggregate_metric_list is None):
                results[group_or_task][" "] = " "
                show_group_table = _apply_group_postprocess(
                    group_obj,
                    group_or_task,
                    results,
                    versions,
                    show_group_table,
                    _task_aggregation_list,
                    aggregate_metric_list=aggregate_metric_list,
                    **hook_kwargs,
                )
                continue

            if "aggregate_metric_list" in group_config:
                agg_metric_list = group_config["aggregate_metric_list"]

            show_group_table = show_group_table | bool(group_config["aggregate_metric_list"])

            task_list = _task_aggregation_list[group_or_task]

            metric_list = list({key for task in task_list for key in results[task].keys() if "_stderr" not in key and key not in ["task", "alias", "samples", "sample_counts"]})
            for metric in metric_list:
                stderr = "_stderr,".join(metric.split(","))

                # gather metrics, sizes, and stderrs from subtasks
                metrics = [results[task][metric] for task in task_list if metric in results[task]]  # TODO: copy?
                stderrs = [results[task][stderr] for task in task_list if stderr in results[task]]
                # weight each subtask by the number of items behind this metric, which can be fewer than its documents
                sizes = [results[task].get("sample_counts", {}).get(metric, results[task]["samples"]) for task in task_list if metric in results[task]]

                for metric_config in agg_metric_list:
                    for filter_name in metric_config["filter_list"]:
                        # if metric != ",".join([metric_config["metric"], filter_name]):
                        #     continue
                        if metric_config["metric"] not in metric:
                            continue

                        # compute group's pooled metric and stderr
                        if metric_config["aggregation"] == "mean":
                            aggregate_fn = aggregate_subtask_metrics
                        elif callable(metric_config["aggregation"]):
                            aggregate_fn = metric_config["aggregation"]
                        else:
                            raise ValueError(f"Currently, only 'mean' is supported for automatically aggregating scores across groups' subtasks. Got '{metric_config['aggregation']}' for group '{group_or_task}'")

                        results[group_or_task][metric] = aggregate_fn(
                            metrics,
                            sizes,
                            metric_config["weight_by_size"],
                        )
                        # TODO: calculate groups' metrics using arbitrary agg fns
                        if "N/A" in stderrs:
                            results[group_or_task][stderr] = "N/A"
                        else:
                            # NOTE: this assumes we are using the mean to aggregate. There are warnings about this elsewhere
                            results[group_or_task][stderr] = pooled_sample_stderr(stderrs, sizes)
                        results[group_or_task].setdefault("sample_counts", {})[metric] = sum(sizes)

                results[group_or_task]["samples"] = sum(results[task]["samples"] for task in task_list if "samples" in results[task])
                group_metadata = group_config.get("metadata", None)
                if group_metadata is not None:
                    versions[group_or_task] = group_metadata.get("version", None)

            show_group_table = _apply_group_postprocess(
                group_obj,
                group_or_task,
                results,
                versions,
                show_group_table,
                _task_aggregation_list,
                aggregate_metric_list=aggregate_metric_list,
                **hook_kwargs,
            )
    # print(results)
    return results, versions, show_group_table, task_aggregation_list


@positional_deprecated
def find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    """
    Search upward in the directory tree to a maximum of three layers
    to find and return the package root (containing the 'tests' folder)
    """
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(f"Unable to find package root within {max_layers} upwards" + f"of {start_path}")


@positional_deprecated
def run_task_tests(task_list: List[str]):
    """
    Find the package root and run the tests for the given tasks
    """
    import pytest

    package_root = find_test_root(start_path=pathlib.Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(f"Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}")


def compute_baseline_comparison(
    current_scores: List[float],
    baseline_scores: List[float],
    baseline_name: str,
) -> Dict[str, Any]:
    """Compute paired t-test comparison between current and baseline."""
    result = paired_ttest(current_scores, baseline_scores)
    result["baseline_name"] = baseline_name
    result["baseline_mean"] = sum(baseline_scores) / len(baseline_scores) if baseline_scores else float("nan")
    result["current_mean"] = sum(current_scores) / len(current_scores) if current_scores else float("nan")
    return result

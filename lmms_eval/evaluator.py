import collections
import copy
import itertools
import json
import os
import random
import re
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

import lmms_eval.api
import lmms_eval.api.metrics
import lmms_eval.api.registry
from lmms_eval import models
from lmms_eval.api.instance import Instance, unwrap_generation_output
from lmms_eval.api.model import lmms
from lmms_eval.api.reasoning import parse_reasoning_tags_config, strip_reasoning_tags
from lmms_eval.api.task import Task
from lmms_eval.baselines import (
    BASELINE_REGISTRY,
    get_baseline_display_name,
    load_baseline,
)
from lmms_eval.caching.response_cache import ResponseCache
from lmms_eval.evaluator_utils import (
    compute_baseline_comparison,
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lmms_eval.llm_judge.launcher import get_launcher
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.models.model_utils.efficiency_metrics import build_efficiency_summary
from lmms_eval.models.model_utils.usage_metrics import (
    is_budget_exceeded,
    reset_usage_metrics,
    set_budget,
    set_task_context,
    summarize_usage_metrics,
)
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import (
    create_iterator,
    get_datetime_str,
    get_git_branch_name,
    get_git_commit_hash,
    get_lmms_eval_version_string,
    handle_non_serializable,
    hash_string,
    is_multimodal_content,
    positional_deprecated,
    run_task_tests,
    simple_parse_args_string,
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    launcher_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    offset: int = 0,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = get_datetime_str(),
    distributed_executor_backend: str = "accelerate",
    cli_args=None,
    force_simple: bool = False,
    repeats: int = 1,
    baseline: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lmms_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        Directory for response-level caching (SQLite + JSONL). `None` to disable.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param offset: int, optional
        Start evaluation from this dataset index for each task.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param repeats: int
        Number of repeated generations per question for k-samples stability metrics.
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.
    :param distributed_executor_backend: str
        The backend to use for distributed execution, `accelerate` or `torchrun`. Defaults to "accelerate" for the `accelerate` library.
    :return
        Dictionary of results
    """
    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    assert distributed_executor_backend in {"accelerate", "torchrun"}, f"Invalid distributed executor backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'."

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning("generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""

    if launcher_args is not None:
        launcher_args = simple_parse_args_string(launcher_args)
        launcher_name = launcher_args.pop("name")
        eval_launcher = get_launcher(launcher_name)(**launcher_args)
    else:
        eval_launcher = None

    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=model)

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = models.get_model(model, force_simple).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    elif isinstance(model, lmms_eval.api.model.lmms):
        lm = model
    task_type = "simple" if lm.is_simple else "chat"
    task_dict = get_task_dict(tasks, task_manager, task_type)

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if predict_only:
                    eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                    else:
                        eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                # Handle repeated generations for model stability measurement (k-samples mode)
                if repeats > 1:
                    default_repeats = task_obj.get_config("repeats") or 1
                    eval_logger.info(f"[Model Stability] Setting repeats={repeats} for {task_name} (was: {default_repeats})")
                    task_obj.set_config(key="repeats", value=repeats)

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    from lmms_eval.models.model_utils.gen_metrics import reset_logged_metrics

    reset_logged_metrics()
    reset_usage_metrics()
    if max_tokens is not None:
        set_budget(max_tokens=max_tokens)

    # Getting the rank settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    response_cache = None
    if use_cache is not None:
        _FUNC_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")

        task_fingerprints = {}
        for tname, tobj in task_dict.items():
            if hasattr(tobj, "dump_config"):
                cfg_str = json.dumps(tobj.dump_config(), sort_keys=True, default=str)
                cfg_str = _FUNC_ADDR_RE.sub(">", cfg_str)
                task_fingerprints[tname] = hash_string(cfg_str)[:16]

        if isinstance(model_args, dict):
            model_args_fp = json.dumps(model_args, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
        elif isinstance(model_args, str):
            try:
                parsed_model_args = simple_parse_args_string(model_args)
            except Exception:
                parsed_model_args = model_args
            if isinstance(parsed_model_args, dict):
                model_args_fp = json.dumps(parsed_model_args, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
            else:
                model_args_fp = str(model_args)
        else:
            model_args_fp = str(model_args)

        model_fp = f"{model}|{model_args_fp}"
        model_hash = hash_string(model_fp)[:16]
        cache_dir = os.path.join(use_cache, model_hash)
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, f"rank{global_rank}.db")
        audit_path = os.path.join(cache_dir, f"rank{global_rank}.jsonl")
        response_cache = ResponseCache(db_path=db_path, audit_path=audit_path, model_fingerprint=model_fp, task_fingerprints=task_fingerprints)
        eval_logger.info(f"ResponseCache initialized: {db_path}")

    try:
        results = evaluate(
            lm=lm,
            task_dict=task_dict,
            limit=limit,
            offset=offset,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            bootstrap_iters=bootstrap_iters,
            write_out=write_out,
            log_samples=True if predict_only else log_samples,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            verbosity=verbosity,
            distributed_executor_backend=distributed_executor_backend,
            cli_args=cli_args,
            eval_server_launcher=eval_launcher,
            response_cache=response_cache,
        )
    finally:
        if response_cache is not None:
            stats = response_cache.get_stats()
            eval_logger.info(f"ResponseCache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['skipped_non_deterministic']} skipped, hit rate: {stats['hit_rate']:.1%}")
            response_cache.close()

    if global_rank == 0:
        from lmms_eval.models.model_utils.gen_metrics import summarize_logged_metrics

        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available TODO: add model info
        # if isinstance(lm, lmms_eval.models.huggingface.HFLM):
        #     results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "offset": offset,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        # Store full resolved CLI args for reproducibility
        if cli_args is not None:
            resolved = {}
            for key, value in vars(cli_args).items():
                try:
                    json.dumps(value)
                    resolved[key] = value
                except (TypeError, ValueError):
                    resolved[key] = str(value)
            results["config"]["resolved_cli_args"] = resolved

        results["git_hash"] = get_git_commit_hash()
        results["git_branch"] = get_git_branch_name()
        results["lmms_eval_version"] = get_lmms_eval_version_string()
        results["date"] = datetime_str
        throughput_summary = summarize_logged_metrics()
        if throughput_summary:
            results["throughput"] = throughput_summary
        usage_summary = summarize_usage_metrics()
        results["usage"] = usage_summary
        efficiency_summary = build_efficiency_summary(results)
        if efficiency_summary:
            results["efficiency"] = efficiency_summary
        # add_env_info(results)  # additional environment info to results
        # add_tokenizer_info(results, lm)  # additional info about tokenizer

        # Baseline comparison (paired t-test)
        if baseline:
            baseline_display_name = get_baseline_display_name(baseline)

            for task_name in results.get("results", {}).keys():
                try:
                    baseline_scores_dict, baseline_agg = load_baseline(baseline, task_name)
                    # Extract current scores from samples
                    if "samples" in results and task_name in results["samples"]:
                        current_samples = results["samples"][task_name]
                        # Get score_key from task config, default to "score"
                        task_config = results.get("configs", {}).get(task_name, {})
                        score_key = task_config.get("score_key", "score")

                        current_scores = []
                        baseline_scores = []
                        for sample in current_samples:
                            doc_id = sample.get("doc_id")
                            if doc_id in baseline_scores_dict:
                                # Extract score: first try exact score_key, then search for *_score fields
                                score = None
                                if score_key in sample:
                                    val = sample[score_key]
                                    if isinstance(val, (int, float)):
                                        score = float(val)
                                    elif isinstance(val, dict) and "score" in val:
                                        score = float(val["score"])
                                # Fallback: search for fields ending with "_score" (e.g., videomme_perception_score)
                                if score is None:
                                    for key in sample:
                                        if key.endswith("_score") and key != score_key:
                                            val = sample[key]
                                            if isinstance(val, (int, float)):
                                                score = float(val)
                                                break
                                            elif isinstance(val, dict) and "score" in val:
                                                score = float(val["score"])
                                                break
                                if score is not None:
                                    current_scores.append(score)
                                    baseline_scores.append(baseline_scores_dict[doc_id])

                        if current_scores and baseline_scores:
                            comparison = compute_baseline_comparison(current_scores, baseline_scores, baseline_display_name)
                            task_results = results["results"][task_name]
                            task_results["paired_baseline"] = comparison["baseline_name"]
                            task_results["paired_baseline_score"] = comparison["baseline_mean"] * 100
                            task_results["paired_ci_lower"] = comparison["ci_lower"] * 100
                            task_results["paired_ci_upper"] = comparison["ci_upper"] * 100
                            task_results["paired_pvalue"] = comparison["p_value"]
                            eval_logger.info(f"[Baseline] {task_name}: diff={comparison['mean_diff']*100:.2f}%, p={comparison['p_value']:.4f}")
                        else:
                            eval_logger.debug(f"[Baseline] Skipping {task_name}: no valid scores found with score_key='{score_key}'")
                except Exception as e:
                    eval_logger.warning(f"[Baseline] Failed for {task_name}: {e}")

        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


def _run_generate_until_agentic(
    lm,
    requests: list[Instance],
    agentic_trace_mode: str = "basic",
    response_cache: Optional[ResponseCache] = None,
) -> list[str]:
    responses: list[str] = []

    for req in requests:
        (
            current_context,
            generation_kwargs,
            current_doc_to_visual,
            doc_to_text,
            doc_id,
            task_name,
            split,
        ) = req.args

        if not callable(doc_to_text):
            raise ValueError("generate_until_agentic requires callable doc_to_text")

        max_agentic_steps = int(generation_kwargs.get("max_agentic_steps", 12))
        base_generation_kwargs = copy.deepcopy(generation_kwargs)
        base_generation_kwargs.pop("max_agentic_steps", None)

        model_outputs: list[str] = []
        previous_round_info = None
        final_response = ""
        full_round_trace: list[dict] = []

        for round_idx in range(max_agentic_steps):
            round_input_context = current_context
            if getattr(lm, "is_simple", False):
                single_req = Instance(
                    request_type="generate_until",
                    arguments=(current_context, copy.deepcopy(base_generation_kwargs), current_doc_to_visual, doc_id, task_name, split),
                    idx=0,
                    metadata=req.metadata,
                )
            else:
                current_doc = lm.task_dict[task_name][split][doc_id]

                def _agentic_doc_to_messages(_doc):
                    visuals = current_doc_to_visual(_doc)
                    if visuals is None:
                        visuals = []
                    content = []
                    for visual in visuals:
                        if isinstance(visual, dict):
                            content.append({"type": "audio", "url": visual})
                        elif isinstance(visual, str):
                            content.append({"type": "video", "url": visual})
                        else:
                            content.append({"type": "image", "url": visual})
                    content.append({"type": "text", "text": current_context})
                    return [{"role": "user", "content": content}]

                single_req = Instance(
                    request_type="generate_until",
                    arguments=(current_context, _agentic_doc_to_messages, copy.deepcopy(base_generation_kwargs), doc_id, task_name, split),
                    idx=0,
                    metadata=req.metadata,
                )
            if response_cache is not None:
                current_raw_output = response_cache.execute(lm, "generate_until", [single_req])[0]
            else:
                current_raw_output = lm.generate_until([single_req])[0]
            current_output, _ = unwrap_generation_output(current_raw_output)
            model_outputs.append(current_output)
            final_response = current_output

            step_payload = doc_to_text(
                lm.task_dict[task_name][split][doc_id],
                previous_output=model_outputs,
                round_idx=round_idx + 1,
                previous_round_info=previous_round_info,
            )

            if isinstance(step_payload, tuple) and len(step_payload) == 5:
                visuals, next_context, terminal_signal, updated_outputs, next_round_info = step_payload
                if updated_outputs is not None:
                    model_outputs = list(updated_outputs)
                    if model_outputs:
                        final_response = model_outputs[-1]
                previous_round_info = next_round_info

                if agentic_trace_mode == "full":
                    round_record = {
                        "round_idx": round_idx + 1,
                        "round_input": round_input_context,
                        "model_output": current_output,
                        "terminal": bool(terminal_signal),
                    }
                    if isinstance(next_round_info, dict):
                        round_record["state"] = next_round_info.get("state")
                        round_record["tool_result"] = next_round_info.get("last_tool_result")
                        round_record["tool_calls"] = next_round_info.get("tool_calls")
                        round_record["valid_tool_calls"] = next_round_info.get("valid_tool_calls")
                        round_record["invalid_steps"] = next_round_info.get("invalid_steps")
                    if next_context is not None:
                        round_record["next_input"] = next_context
                    full_round_trace.append(round_record)

                if terminal_signal:
                    break

                if next_context is not None:
                    current_context = next_context
                if visuals is not None:
                    current_doc_to_visual = lambda _doc, _visuals=visuals: _visuals
            elif isinstance(step_payload, str):
                current_context = step_payload
            else:
                break

        if previous_round_info is not None and not (isinstance(final_response, str) and final_response.strip().startswith("{")):
            state = previous_round_info.get("state", {}) if isinstance(previous_round_info, dict) else {}
            valid_tool_calls = float(previous_round_info.get("valid_tool_calls", previous_round_info.get("tool_calls", 0))) if isinstance(previous_round_info, dict) else 0.0
            invalid_steps = float(previous_round_info.get("invalid_steps", 0.0)) if isinstance(previous_round_info, dict) else 0.0
            fallback_payload = {
                "success": False,
                "error": "max_agentic_steps_reached",
                "tool_calls": float(previous_round_info.get("tool_calls", 0)) if isinstance(previous_round_info, dict) else 0.0,
                "valid_tool_calls": valid_tool_calls,
                "invalid_steps": invalid_steps,
                "state": state,
                "last_model_output": final_response,
                "trace": model_outputs,
            }
            if isinstance(state, dict):
                for key in ["cash", "days_elapsed", "inventory", "mobile_data_working"]:
                    if key in state:
                        fallback_payload[key] = state[key]
            final_response = json.dumps(fallback_payload, ensure_ascii=False)

        if agentic_trace_mode == "full":
            try:
                parsed_response = json.loads(final_response) if isinstance(final_response, str) else None
                if isinstance(parsed_response, dict):
                    parsed_response["agentic_trace_mode"] = "full"
                    parsed_response["agentic_rounds"] = full_round_trace
                    final_response = json.dumps(parsed_response, ensure_ascii=False)
            except (TypeError, json.JSONDecodeError):
                pass

        responses.append(final_response)

    return responses


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit: Optional[int] = None,
    offset: int = 0,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    distributed_executor_backend: str = "accelerate",
    eval_server_launcher: Optional[Union[str, Callable]] = None,
    cli_args=None,
    response_cache: Optional[ResponseCache] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param offset: int, optional
        Start evaluation from this dataset index for each task.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param distributed_executor_backend: str
        The backend to use for distributed execution, `accelerate` or `torchrun`. Defaults to "accelerate" for the `accelerate` library.
    :return
        Dictionary of results
    """

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store the ordering of tasks and groups
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    # Getting the rank settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    eval_logger.info(f"Running on rank {global_rank} (local rank {local_rank})")

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    name_to_task = {}
    if not log_samples:
        if not all("bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys() for task_output in eval_tasks):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    if distributed_executor_backend == "accelerate" and not hasattr(lm, "accelerator"):
        lm.accelerator = Accelerator()

    for task_output in eval_tasks:
        task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args

        name_to_task[task_name] = task

        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"
        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        if ("group_alias" in configs[task_name]) and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            offset=offset,
            rank=global_rank,
            world_size=world_size,
            cache_requests=cache_requests,  # later we will add them
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
        eval_logger.debug(f"Task: {task_output.task_name}; number of requests on this rank: {len(task._instances)}")
        if write_out:
            eval_logger.warning(
                "DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. "
                "Use --log_samples instead for saving model outputs and debugging. "
                "The write_out flag only prints the first few documents and impacts performance."
            )
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            elif distributed_executor_backend == "torchrun":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = torch.zeros(world_size * 1, dtype=instances_rnk.dtype, device=lm.device)
                dist.all_gather_into_tensor(gathered_item, instances_rnk)
                gathered_item = gathered_item.cpu().detach().numpy().tolist()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = task.instances[0].request_type
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    ### Run LMM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info("Running {} requests".format(reqtype))
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model (with optional response cache)
        if reqtype == "generate_until_agentic":
            trace_mode = "basic"
            if cli_args is not None:
                trace_mode = getattr(cli_args, "agentic_trace_mode", "basic")
            resps = _run_generate_until_agentic(
                lm,
                cloned_reqs,
                agentic_trace_mode=trace_mode,
                response_cache=response_cache,
            )
        elif response_cache is not None:
            resps = response_cache.execute(lm, reqtype, cloned_reqs)
        else:
            resps = getattr(lm, reqtype)(cloned_reqs)

        for x, req in zip(resps, cloned_reqs):
            text, tc = unwrap_generation_output(x)
            req.resps.append(text)
            req.token_counts.append(tc)

        if is_budget_exceeded():
            eval_logger.warning("Token budget reached after '{}' requests. Skipping remaining request types.", reqtype)
            break

        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                lm.accelerator.wait_for_everyone()
            elif distributed_executor_backend == "torchrun":
                dist.barrier()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

    # Cleaning lm's cuda memory if you are launching llm as judge in local
    lm.clean()
    RANK = global_rank
    WORLD_SIZE = world_size
    if eval_server_launcher is not None and RANK == 0:
        eval_server_launcher.launch()

    if world_size > 1:
        if distributed_executor_backend == "accelerate":
            lm.accelerator.wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()
        set_task_context(task_output.task_name)

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            # Resolve reasoning tags for this task
            cli_reasoning_tags = getattr(cli_args, "reasoning_tags", None) if cli_args else None
            task_reasoning_tags = getattr(task.config, "reasoning_tags", None)
            reasoning_tags = parse_reasoning_tags_config(cli_value=cli_reasoning_tags, task_value=task_reasoning_tags)

            if cli_args is not None and not cli_args.process_with_media:
                doc_iterator = create_iterator(
                    enumerate(task.eval_docs_no_media),
                    rank=RANK,
                    limit=int(limit) if limit else None,
                    world_size=WORLD_SIZE,
                    offset=offset,
                )
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE, offset=offset)
            doc_iterator_for_counting = (
                create_iterator(
                    range(len(task.test_docs())),
                    rank=RANK,
                    limit=limit,
                    world_size=WORLD_SIZE,
                    offset=offset,
                )
                if task.has_test_docs()
                else create_iterator(
                    range(len(task.validation_docs())),
                    rank=RANK,
                    limit=limit,
                    world_size=WORLD_SIZE,
                    offset=offset,
                )
            )
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=total_docs, desc="Postprocessing", disable=(RANK != 0))
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]

                # Strip reasoning tags before scoring
                if reasoning_tags is not None:
                    for req in requests:
                        raw_resp = req.filtered_resps[filter_key]
                        req.raw_filtered_resps[filter_key] = raw_resp
                        if isinstance(raw_resp, str):
                            req.filtered_resps[filter_key] = strip_reasoning_tags(raw_resp, reasoning_tags)
                        elif isinstance(raw_resp, list):
                            req.filtered_resps[filter_key] = [strip_reasoning_tags(r, reasoning_tags) if isinstance(r, str) else r for r in raw_resp]

                metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])

                # For stability metrics: compute per-sample scores when repeats > 1
                repeats = task.config.repeats if hasattr(task, "config") and hasattr(task.config, "repeats") else 1
                if repeats > 1 and len(requests) == repeats:
                    # Compute per-sample scores by calling process_results for each sample individually
                    per_sample_scores = {}
                    for req in requests:
                        sample_metrics = task.process_results(doc, [req.filtered_resps[filter_key]])
                        for metric_name, value in sample_metrics.items():
                            if metric_name not in per_sample_scores:
                                per_sample_scores[metric_name] = []
                            per_sample_scores[metric_name].append(value)
                    # Store per-sample scores grouped by doc_id
                    for metric_name, scores in per_sample_scores.items():
                        task_output.per_sample_metrics[(metric_name, filter_key)].append(scores)

                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        if not is_multimodal_content(value):
                            saved_doc[key] = value
                    filtered_arguments = []
                    for req in requests:
                        # check if req.args is a list of tuples, and each item in the list is a serializable object
                        for value in req.args:
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                filtered_arguments.append(value)
                            # else:
                            #     filtered_arguments.append(_handle_non_serializable(value))

                    per_sample_tc = []
                    for req in requests:
                        if req.token_counts:
                            tc = req.token_counts[0]
                            per_sample_tc.append(tc.to_dict() if tc is not None else None)
                        else:
                            per_sample_tc.append(None)

                    example = {
                        "doc_id": doc_id,
                        "doc": saved_doc,
                        "target": target,
                        "arguments": filtered_arguments,
                        "resps": [req.raw_filtered_resps.get(filter_key, req.resps) for req in requests],
                        "filtered_resps": [req.filtered_resps[filter_key] for req in requests],
                        "token_counts": per_sample_tc,
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)
                pbar.update(1)

            pbar.close()
        set_task_context(None)

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = []
                for sample in task_output.logged_samples:
                    per_rank_samples.append(sample)

                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))

            # then collect metrics across all ranks
            # All ranks must iterate over metric keys in the SAME order,
            # otherwise gather_object calls will misalign values between keys.
            # Gather all keys, merge, sort, and broadcast a canonical order.
            # this important when returning many keys from a benchmark, to avoid misalignments between ranks.
            all_metric_keys = list(task_output.sample_metrics.keys())
            gathered_keys = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(
                obj=all_metric_keys,
                object_gather_list=gathered_keys,
                dst=0,
            )

            if RANK == 0:
                all_keys_set = set()
                for rank_keys in gathered_keys:
                    if rank_keys:
                        all_keys_set.update(rank_keys)
                canonical_keys = sorted(all_keys_set, key=lambda x: str(x))
            else:
                canonical_keys = None

            broadcast_list = [canonical_keys] if RANK == 0 else [None]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)
            canonical_keys = broadcast_list[0]

            for metrics in canonical_keys:
                if metrics in task_output.sample_metrics:
                    pre_gather = task_output.sample_metrics[metrics]
                else:
                    pre_gather = []

                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=pre_gather,
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))

            # gather per_sample_metrics for stability metrics (same canonical ordering)
            all_ps_keys = list(task_output.per_sample_metrics.keys())
            gathered_ps_keys = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(
                obj=all_ps_keys,
                object_gather_list=gathered_ps_keys,
                dst=0,
            )

            if RANK == 0:
                all_ps_set = set()
                for rank_keys in gathered_ps_keys:
                    if rank_keys:
                        all_ps_set.update(rank_keys)
                canonical_ps_keys = sorted(all_ps_set, key=lambda x: str(x))
            else:
                canonical_ps_keys = None

            broadcast_ps = [canonical_ps_keys] if RANK == 0 else [None]
            torch.distributed.broadcast_object_list(broadcast_ps, src=0)
            canonical_ps_keys = broadcast_ps[0]

            for metrics in canonical_ps_keys:
                if metrics in task_output.per_sample_metrics:
                    pre_gather = task_output.per_sample_metrics[metrics]
                else:
                    pre_gather = []

                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=pre_gather,
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.per_sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))

        dist.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        if eval_server_launcher is not None:
            eval_server_launcher.clean()
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
            task_output.calculate_clt_aggregate_metric()
            task_output.calculate_stability_metrics()
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if m in _higher_is_better and _higher_is_better[m] is not None and _higher_is_better[m] != h:
                            eval_logger.warning(f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None.")
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(group_agg.items())} if (bool(group_agg) & show_group_table) else {}),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }
        if log_samples:
            results_dict["samples"] = dict(samples)
    else:
        results_dict = None

    if WORLD_SIZE > 1:
        # if muti-gpu, wait for all processes to finish
        if distributed_executor_backend == "accelerate":
            # this should work for torchrun as well since it internally calls torch.distributed.barrier()
            Accelerator().wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()
        else:
            raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

    return results_dict


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args

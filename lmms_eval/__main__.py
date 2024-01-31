import os
import yaml
import sys
import copy
import json
import logging
import argparse
import numpy as np

from accelerate import Accelerator
from pathlib import Path
from typing import Union
import hashlib
import wandb

from lmms_eval import evaluator, utils
from lmms_eval.tasks import initialize_tasks, include_path, get_task_dict
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.api.wandb import log_eval_result


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument("--batch_size", type=str, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--show_task_to_terminal",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles",
    )
    args = parser.parse_args()
    return args


def cli_evaluate(args: Union[argparse.Namespace, None] = None, wandb_run=None) -> None:
    if args is None:
        args = parse_eval_args()

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # run each config
    for args in args_list:
        results = cli_evaluate_single(args)
        results_list.append(results)
    # print results
    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            print_results(args, results)
            # Add W&B logic
            if args.wandb_args:
                wandb_results = copy.deepcopy(results)
                log_eval_result(wandb_run, wandb_results)


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initialize_tasks(args.verbosity)

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format(f"\n - ".join(sorted(ALL_TASKS))))
        sys.exit()
    elif args.tasks == "list_tasks_num":
        log_message = (
            "\n" + "=" * 70 + "\n" + "\n\tYou are trying to check all the numbers in each task." + "\n\tThis action will download the complete dataset." + "\n\tIf the results are not clear initially, call this again." + "\n\n" + "=" * 70
        )
        eval_logger.info(log_message)
        task_dict = get_task_dict([task for task in sorted(ALL_TASKS)], model_name=args.model)
        for task_name in task_dict.keys():
            task_obj = task_dict[task_name]
            if type(task_obj) == tuple:
                group, task_obj = task_obj
                if task_obj is None:
                    continue
            eval_logger.info(f"\nTask : {task_obj.config.task}\n - #num : {len(task_obj.test_docs()) if task_obj.has_test_docs() else task_obj.validation_docs()}")
        sys.exit()
    else:
        tasks_list = args.tasks.split(",")
        eval_logger.info(f"Evaluating on {len(tasks_list)} tasks.")
        task_names = utils.pattern_match(tasks_list, ALL_TASKS)
        task_missing = [task for task in tasks_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}. Try `lmms-eval --tasks list` for list of available tasks",
            )
            # eval_logger.warn(f"Tasks {missing} were not found. Try `lmms-eval --tasks list` for list of available tasks.")

    eval_logger.info(f"Selected Tasks: {task_names}")

    # set datetime before evaluation
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        check_integrity=args.check_integrity,
        show_task_to_terminal=args.show_task_to_terminal,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
    )

    if args.output_path:
        hash_input = f"{args.model_args}".encode("utf-8")
        hash_output = hashlib.sha256(hash_input).hexdigest()[:6]
        path = Path(args.output_path)
        path = path.expanduser().resolve().joinpath(f"{args.model}").joinpath(f"model_args_{hash_output}").joinpath(f"{datetime_str}")
        path.mkdir(parents=True, exist_ok=True)
        assert path.is_dir(), f"Output path {path} is not a directory"

        output_path_file = path.joinpath("results.json")
        if output_path_file.exists():
            eval_logger.warning(f"Output file {output_path_file} already exists and will be overwritten.")

    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = f"{task_name}_{args.log_samples_suffix}"
                    filename = path.joinpath(f"{output_name}.json")
                    # Structure the data with 'args' and 'logs' keys
                    data_to_dump = {"args": vars(args), "config": config, "logs": sorted(samples[task_name], key=lambda x: x["doc_id"])}  # Convert Namespace to dict
                    samples_dumped = json.dumps(data_to_dump, indent=4, default=_handle_non_serializable)
                    filename.open("w").write(samples_dumped)
                    eval_logger.info(f"Saved samples to {filename}")

        return results
    return None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, " f"batch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    args = parse_eval_args()
    args_list = []
    results_list = []
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator
    accelerator = Accelerator()
    all_args_dict = vars(args)
    wandb_run = None

    if accelerator.is_main_process:
        # initialize a W&B run only on rank 0
        wandb_args_dict = utils.simple_parse_args_string(args.wandb_args)
        if wandb_args_dict:
            if "name" not in wandb_args_dict:
                if "config" not in all_args_dict:
                    # use the model name and task names as run name
                    task_names = args.tasks.replace(",", "_")
                    wandb_args_dict["name"] = f"{args.model}_{task_names}_{args.log_samples_suffix}"
                    if args.num_fewshot:
                        wandb_args_dict["name"] += f"_{args.num_fewshot}shot"
                else:
                    # use the name of the config file as run name
                    wandb_args_dict["name"] = all_args_dict["config"].split("/")[-1].split(".")[0]
            wandb_run = wandb.init(**wandb_args_dict)
        is_main_process = True
    else:
        is_main_process = False

    # run each config
    args.is_main_process = is_main_process
    for args in args_list:
        results = cli_evaluate(args, wandb_run)
        results_list.append(results)

    if is_main_process and wandb_run is not None:
        wandb_run.finish()

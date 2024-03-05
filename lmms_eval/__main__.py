import os
import yaml
import sys
import copy
import json
import logging
import traceback
import argparse
import torch
import numpy as np
import datetime

import warnings
import traceback

warnings.simplefilter("ignore", category=DeprecationWarning)

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from pathlib import Path
from typing import Union
import hashlib

from lmms_eval import evaluator, utils
from lmms_eval.tasks import initialize_tasks, include_path, get_task_dict
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.logging_utils import WandbLogger
from lmms_eval.utils import PathFormatter


eval_logger = logging.getLogger("lmms-eval")


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
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
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


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()

    # Check if no arguments were passed after parsing
    if len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `lmms-eval --model llava --model_path liuhaotian/llava-v1.6-7b --tasks okvqa` │")
        print("│ Use `lmms-eval --help` for more information.                                  │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    set_loggers(args)
    eval_logger = logging.getLogger("lmms-eval")
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # initialize Accelerator
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    if accelerator.is_main_process:
        is_main_process = True
    else:
        is_main_process = False

    for args in args_list:
        try:
            if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
                wandb_logger = WandbLogger(args)

            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            accelerator.wait_for_everyone()
            if is_main_process and args.wandb_args:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.wandb_log_samples and samples is not None:
                    wandb_logger.log_eval_samples(samples)

                wandb_logger.finish()

        except Exception as e:
            traceback.print_exc()
            eval_logger.error(f"Error during evaluation: {e}")
            traceback.print_exc()
            results_list.append(None)

    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            print_results(args, results)


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    eval_logger = logging.getLogger("lmms-eval")
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
    elif args.tasks == "list_with_num":
        log_message = (
            "\n" + "=" * 70 + "\n" + "\n\tYou are trying to check all the numbers in each task." + "\n\tThis action will download the complete dataset." + "\n\tIf the results are not clear initially, call this again." + "\n\n" + "=" * 70
        )
        eval_logger.info(log_message)
        task_dict = get_task_dict([task for task in sorted(ALL_TASKS)], model_name="llava")
        for task_name in task_dict.keys():
            task_obj = task_dict[task_name]
            if type(task_obj) == tuple:
                group, task_obj = task_obj
                if task_obj is None:
                    continue
            eval_logger.info(f"\nTask : {task_obj.config.task}\n - #num : {len(task_obj.test_docs()) if task_obj.has_test_docs() else len(task_obj.validation_docs())}")
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
    if args.output_path:
        hash_input = f"{args.model_args}".encode("utf-8")
        hash_output = hashlib.sha256(hash_input).hexdigest()[:6]
        path = Path(args.output_path)
        path = path.expanduser().resolve().joinpath(f"{args.model}").joinpath(f"model_args_{hash_output}").joinpath(f"{datetime_str}_{args.log_samples_suffix}")
        args.output_path = path

    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

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
        cli_args=args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        if args.output_path:
            args.output_path.mkdir(parents=True, exist_ok=True)
            result_file_path = path.joinpath("results.json")
            if result_file_path.exists():
                eval_logger.warning(f"Output file {result_file_path} already exists and will be overwritten.")

            result_file_path.open("w").write(dumped)
            if args.log_samples:
                for task_name, config in results["configs"].items():
                    filename = args.output_path.joinpath(f"{task_name}.json")
                    # Structure the data with 'args' and 'logs' keys
                    data_to_dump = {"args": vars(args), "model_configs": config, "logs": sorted(samples[task_name], key=lambda x: x["doc_id"])}  # Convert Namespace to dict
                    samples_dumped = json.dumps(data_to_dump, indent=4, default=_handle_non_serializable)
                    filename.open("w").write(samples_dumped)
                    eval_logger.info(f"Saved samples to {filename}")

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


def set_loggers(args):
    eval_logger = logging.getLogger("lmms-eval")
    ch = logging.StreamHandler()
    formatter = PathFormatter("%(asctime)s [%(pathname)s:%(lineno)d] %(levelname)s %(message)s", "%m-%d %H:%M:%S", timezone=args.timezone)
    ch.setFormatter(formatter)
    eval_logger.addHandler(ch)


if __name__ == "__main__":
    cli_evaluate()

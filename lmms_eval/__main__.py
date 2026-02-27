import sys

# CLI dispatcher which handles subcommands (tasks, models, eval, ui, ...)
# as well as the legacy flat-args form (--model X --tasks Y).
if __name__ == "__main__":
    from lmms_eval.cli.dispatch import main

    main()
    sys.exit(0)

import argparse
import datetime
import importlib
import json
import os
import traceback
import warnings
from functools import partial

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

import lmms_eval.tasks
from lmms_eval import evaluator, utils
from lmms_eval.api.metrics import power_analysis
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    get_eval_banner,
    make_table,
    simple_parse_args_string,
)


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        eval_logger.warning(f"Argument requires {max_len} integers or None, separated by '{split_char}'. " "Missing values will be filled with defaults.")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _run_power_analysis(args: argparse.Namespace) -> None:
    """Run power analysis to calculate minimum sample size for detecting a given effect."""
    task_sizes = {}
    if args.tasks and args.tasks not in ["list", "list_groups", "list_tags", "list_subtasks"]:
        task_manager = TaskManager(args.verbosity, include_path=args.include_path)
        task_names = task_manager.match_tasks(args.tasks.split(","))
        for task_name in task_names:
            task_dict = lmms_eval.tasks.get_task_dict([task_name], task_manager)
            for name, task_obj in task_dict.items():
                if hasattr(task_obj, "eval_docs"):
                    task_sizes[name] = len(task_obj.eval_docs)

    result = power_analysis(
        effect_size=args.effect_size,
        std_a=args.std_a,
        std_b=args.std_b,
        alpha=args.alpha,
        power=args.power,
        correlation=args.correlation,
    )

    print("\n" + "=" * 60)
    print("POWER ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Effect size (delta):     {args.effect_size:.1%}")
    print(f"  Std (model A):           {result['std_a']}")
    print(f"  Std (model B):           {result['std_b']}")
    print(f"  Significance level (α):  {args.alpha}")
    print(f"  Desired power (1-β):     {args.power}")
    print(f"  Correlation (ρ):         {args.correlation}")
    print(f"\nResult:")
    print(f"  Minimum sample size:     n = {result['min_n']}")
    print(f"\nInterpretation:")
    print(f"  To detect a {args.effect_size:.1%} difference with {args.power:.0%} power,")
    print(f"  you need at least {result['min_n']} questions in your benchmark.")

    if task_sizes:
        print(f"\n" + "-" * 60)
        print("TASK ANALYSIS")
        print("-" * 60)
        for task_name, n_samples in task_sizes.items():
            task_result = power_analysis(
                effect_size=args.effect_size,
                std_a=args.std_a,
                std_b=args.std_b,
                alpha=args.alpha,
                power=args.power,
                correlation=args.correlation,
                current_n=n_samples,
            )
            status = "✓ Sufficient" if n_samples >= result["min_n"] else "✗ Insufficient"
            print(f"\n  {task_name}:")
            print(f"    Sample size:         n = {n_samples}")
            print(f"    Current power:       {task_result['current_power']:.1%}")
            print(f"    Min detectable Δ:    {task_result['min_detectable_effect']:.1%}")
            print(f"    Status:              {status}")

    print("\n" + "=" * 60 + "\n")


def parse_eval_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        default="",
        help="Path to a yaml file specifying eval arguments. CLI arguments override YAML values.",
    )
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
        "--launcher_args",
        default=None,
        help="String arguments for launcher for local llm as judge, e.g. `tp=8`, if None then no launcher will be used.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
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
        help=("Limit examples per task: use -1 (or omit) for all samples, " "0 < limit < 1 for a fraction of the dataset, and limit >= 1 " "for an absolute sample count."),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start evaluation from this dataset index for each task.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for response-level caching (SQLite + JSONL audit log). "
        "Caches deterministic model responses (temperature=0) for reuse across runs. "
        "Per-rank files created automatically for distributed safety. `None` to disable.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="DEPRECATED: This flag is deprecated and will be removed in a future version. "
        "For debugging, use --log_samples to save all outputs to files. "
        "This flag prints prompts for the first few documents to console, impacting performance.",
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
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
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
        "--reasoning_tags",
        type=str,
        default='[["<think>", "</think>"], ["<analysis>", "</analysis>"]]',
        help="JSON string list of [start_tag, end_tag] pairs used for reasoning extraction.",
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
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument(
        "--process_with_media",
        action="store_true",
        help="Whether you will process you dataset with audio, image. By default set to False" "In case some benchmarks need to be processed with media, set this flag to True.",
    )
    parser.add_argument(
        "--agentic_trace_mode",
        type=str,
        default="basic",
        choices=["basic", "full"],
        help="Controls agentic trace logging level. 'basic' logs compact final trace payload, 'full' logs per-round input/output/state snapshots.",
    )
    parser.add_argument(
        "--force_simple",
        action="store_true",
        help="Force the evaluation to use the simple mode of the models",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch interactive TUI mode for configuration",
    )
    parser.add_argument(
        "-n",
        "--repeats",
        "--num_samples",
        dest="repeats",
        type=int,
        default=1,
        help=("Number of repeated generations per question for model stability " "measurement. Backward-compatible alias: --num_samples. " "When n > 1, enables k-samples " "mode and computes EA, CA, IV, CR metrics."),
    )
    parser.add_argument("--baseline", type=str, default=None, help="Baseline for paired t-test comparison. Accepts: local JSONL path, hf://user/repo, or preset name (e.g., qwen25vl).")

    # Cost & Token Tracking
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum total tokens (input+output+reasoning). Evaluation stops gracefully when exceeded. Disabled by default.",
    )

    # Power Analysis arguments
    parser.add_argument(
        "--power-analysis",
        action="store_true",
        default=False,
        help="Enable power analysis to calculate minimum sample size for detecting a given effect size.",
    )
    parser.add_argument(
        "--effect-size",
        type=float,
        default=0.03,
        help="Minimum effect size to detect (default: 0.03 = 3%%). Used with --power-analysis.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for power analysis (default: 0.05). Used with --power-analysis.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Desired statistical power (default: 0.80). Used with --power-analysis.",
    )
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.5,
        help="Expected correlation between paired samples (default: 0.5). Used with --power-analysis.",
    )
    parser.add_argument(
        "--std-a",
        type=float,
        default=None,
        help="Std deviation of model A scores (estimate from previous eval). Default: 0.5 for binary. Used with --power-analysis.",
    )
    parser.add_argument(
        "--std-b",
        type=float,
        default=None,
        help="Std deviation of model B scores (estimate from previous eval). If not set, assumes equal to --std-a. Used with --power-analysis.",
    )

    args = parser.parse_args()
    return parser, args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    parser, default_args = parse_eval_args()

    # If args were provided, override the defaults
    if args:
        for key, value in vars(args).items():
            setattr(default_args, key, value)

    args = default_args

    # Handle power analysis mode (pre-evaluation planning)
    if getattr(args, "power_analysis", False):
        _run_power_analysis(args)
        sys.exit(0)

    if args.wandb_args:
        if "name" not in args.wandb_args:
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # reset logger
    eval_logger.remove()
    # Configure logger with detailed format including file path, function name, and line number
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity, format=log_format)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args

        # Extract and apply env vars before validation (env is not a CLI arg)
        for config in config_args:
            env_config = config.pop("env", None)
            if env_config:
                if not isinstance(env_config, dict):
                    raise ValueError(f"'env' in config must be a dict, got {type(env_config).__name__}")
                for env_key, env_value in env_config.items():
                    resolved = os.path.expandvars(str(env_value))
                    os.environ[env_key] = resolved
                    eval_logger.info(f"Config env: {env_key}={'*' * min(len(resolved), 8) if any(s in env_key.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) else resolved}")

        # Validate config keys
        valid_keys = {action.dest for action in parser._actions}
        for config in config_args:
            unknown_keys = set(config.keys()) - valid_keys
            if unknown_keys:
                raise ValueError(f"Unknown keys in config file: {sorted(unknown_keys)}. " f"Valid keys are: {sorted(valid_keys - {'help'})}")

        # Determine which CLI args were explicitly provided by the user.
        default_config_args = parser.parse_args([])
        cli_explicit = {}
        for key, value in vars(args).items():
            default_value = getattr(default_config_args, key, None)
            if value != default_value:
                cli_explicit[key] = value

        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(default_config_args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            for key, value in cli_explicit.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator only if not already in a distributed context
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator = None
        is_main_process = torch.distributed.get_rank() == 0
    else:
        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
        if accelerator.is_main_process:
            is_main_process = True
        else:
            is_main_process = False

    for args in args_list:
        try:
            # if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
            #     wandb_logger = WandbLogger()

            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            if accelerator:
                accelerator.wait_for_everyone()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_main_process and args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")
                # wandb_logger.finish()

        except Exception as e:
            if args.verbosity == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                eval_logger.error(f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
                results_list.append(None)

    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            print(f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), " f"limit: {args.limit}, offset: {args.offset}, num_fewshot: {args.num_fewshot}, " f"batch_size: {args.batch_size}")
            print(get_eval_banner(branch=results.get("git_branch"), commit=results.get("git_hash")))
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

    if args.wandb_args:
        wandb_logger.run.finish()


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    eval_logger.info(f"Evaluation tracker args: {evaluation_tracker_args}")

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.write_out:
        eval_logger.warning(
            "DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. "
            "For debugging and analysis, use --log_samples instead, which saves all model "
            "outputs to files without impacting performance. The --write_out flag only prints "
            "the first few documents to console and provides limited debugging value."
        )

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be set to True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub.")

    if args.limit is not None and args.limit != -1:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
    if args.limit is not None and args.limit < 0 and args.limit != -1:
        raise ValueError("--limit must be -1 or non-negative")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" f"{utils.SPACING}Try `lmms-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lmms-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        offset=args.offset,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        distributed_executor_backend="torchrun" if (torch.distributed.is_available() and torch.distributed.is_initialized()) else "accelerate",
        force_simple=args.force_simple,
        launcher_args=args.launcher_args,
        repeats=args.repeats,
        baseline=args.baseline,
        max_tokens=args.max_tokens,
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None

        # Print token usage summary if available
        if results.get("usage") and "total" in results["usage"]:
            u = results["usage"]["total"]
            eval_logger.info(
                "Token Usage - Input: {} | Output: {} | Reasoning: {} | Total: {} | API Calls: {}",
                f"{u['input_tokens']:,}",
                f"{u['output_tokens']:,}",
                f"{u['reasoning_tokens']:,}",
                f"{u['total_tokens']:,}",
                f"{u['n_api_calls']:,}",
            )
            if results["usage"].get("budget_exceeded"):
                eval_logger.warning("Evaluation stopped early: token budget exceeded. Results are partial.")

        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(
            results=results,
            samples=samples if args.log_samples else None,
            datetime_str=datetime_str,
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\n" f"gen_kwargs: ({args.gen_kwargs}),\n" f"limit: {args.limit},\n" f"offset: {args.offset},\n" f"num_fewshot: {args.num_fewshot},\n" f"batch_size: {args.batch_size}")
    print(get_eval_banner(branch=results.get("git_branch"), commit=results.get("git_hash")))
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()

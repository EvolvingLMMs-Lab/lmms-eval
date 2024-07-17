import argparse
import importlib
import os
import yaml
import json
from pathlib import Path
import hashlib

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
import numpy as np
import datetime

from lmms_eval.utils import simple_parse_args_string
import shrinker as shrinker_module


AVAILABEL_SHRINKER = {"embed": "Embed_Shrinker"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shrinker", type=str, help="The type of shrinker you want to use")
    parser.add_argument("--num_items", type=str, help="The number of items you want in your shrinked dataset")
    parser.add_argument("--tasks", type=str, help="The task you want to shrink. Separate each task with comma, will be parsed in to list")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Whether to push the shrinked dataset to hub")
    parser.add_argument("--shrinker_kwargs", type=str, help="In args=xxx,args2=xxx format. Will be parsed into dict")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    shrinker_kwargs = simple_parse_args_string(args.shrinker_kwargs)
    shrinker_name = args.shrinker
    tasks = args.tasks.split(",")
    num_items = args.num_items.split(",")
    assert len(num_items) == 1 or len(num_items) == len(tasks), "Either provide one num items for all task or one num item for each task"
    if len(num_items) == 1:
        num_items = [float(num_items[0])] * len(tasks)
    else:
        num_items = [float(n) for n in num_items]
    push_to_hub = args.push_to_hub
    assert len(num_items) == len(tasks) or len(num_items) == 1, "Either pass in one num_items for whole tasks, or pass in num items for each task"
    assert shrinker_name in AVAILABEL_SHRINKER, f"Unavailable shrinker {shrinker_name}. You can choose from {AVAILABEL_SHRINKER.keys()}"

    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    for idx, task in enumerate(tasks):
        shrinker = getattr(shrinker_module, f"{AVAILABEL_SHRINKER[shrinker_name]}")
        shrinker = shrinker(task=task, num_items=num_items[idx], push_to_hub=push_to_hub, name=shrinker_name, **shrinker_kwargs)
        shrinker.shrink()
        accelerator.wait_for_everyone()

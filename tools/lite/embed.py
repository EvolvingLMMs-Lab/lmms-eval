import argparse
import os

import embedder
import torch.distributed as dist

from lmms_eval.api.registry import ALL_TASKS, GROUP_REGISTRY
from lmms_eval.tasks import (
    ConfigurableTask,
    get_task_dict,
    include_path,
    initialize_tasks,
)
from lmms_eval.utils import simple_parse_args_string


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--tasks", type=str, required=False, default="")
    parser.add_argument("--data_path", type=str, required=False, default="")
    parser.add_argument("--image_folder", type=str, required=False, default="")
    parser.add_argument("--embedder_kwargs", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    embedder_name = args.name
    output_path = args.output_path
    if args.tasks.lower().strip() == "all":
        initialize_tasks()
        for task in list(ALL_TASKS):
            if task in GROUP_REGISTRY:
                ALL_TASKS.remove(task)
        tasks = list(ALL_TASKS)
    else:
        tasks = args.tasks.split(",")

    cached_idx = []
    for idx in range(len(tasks)):
        if os.path.exists(os.path.join(output_path, f"{tasks[idx]}_embed.npy")):
            rank0_print(f"Task {tasks[idx]} exists in cache folder, load from cache")
            cached_idx.append(idx)
    tasks = [tasks[idx] for idx in range(len(tasks)) if idx not in cached_idx]
    rank0_print(f"Tasks : {tasks}")
    embedder_kwargs = simple_parse_args_string(args.embedder_kwargs)

    embedder_cls = getattr(embedder, embedder_name)
    embedder_obj = embedder_cls(name=embedder_name, output_path=output_path, **embedder_kwargs)
    for task in tasks:
        embedder_obj.embed_task(task)

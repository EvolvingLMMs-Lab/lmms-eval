import os, sys
from typing import List, Union, Dict

from lmms_eval import utils

# from lmms_eval import prompts
from lmms_eval.api.task import TaskConfig, Task, ConfigurableTask
from lmms_eval.api.registry import (
    register_task,
    register_group,
    TASK_REGISTRY,
    GROUP_REGISTRY,
    ALL_TASKS,
)

from loguru import logger

eval_logger = logger


def register_configurable_task(config: Dict[str, str]) -> int:
    SubClass = type(
        config["task"] + "ConfigurableTask",
        (ConfigurableTask,),
        {"CONFIG": TaskConfig(**config)},
    )

    if "task" in config:
        task_name = "{}".format(config["task"])
        register_task(task_name)(SubClass)

    if "group" in config:
        if config["group"] == config["task"]:
            raise ValueError("task and group name cannot be the same")
        elif type(config["group"]) == str:
            group_name = [config["group"]]
        else:
            group_name = config["group"]

        for group in group_name:
            register_group(group)(SubClass)

    return 0


def register_configurable_group(config: Dict[str, str]) -> int:
    group = config["group"]
    task_list = config["task"]
    task_names = utils.pattern_match(task_list, ALL_TASKS)
    for task in task_names:
        if (task in TASK_REGISTRY) or (task in GROUP_REGISTRY):
            if group in GROUP_REGISTRY:
                GROUP_REGISTRY[group].append(task)
            else:
                GROUP_REGISTRY[group] = [task]
                ALL_TASKS.add(group)
    return 0


def get_task_name_from_config(task_config: Dict[str, str]) -> str:
    if "dataset_name" in task_config:
        return "{dataset_path}_{dataset_name}".format(**task_config)
    else:
        return "{dataset_path}".format(**task_config)


def include_task_folder(task_dir: str, register_task: bool = True) -> None:
    """
    Calling this function
    """
    for root, subdirs, file_list in os.walk(task_dir):
        # if (subdirs == [] or subdirs == ["__pycache__"]) and (len(file_list) > 0):
        for f in file_list:
            # if "detail" in f:
            #     import pdb;pdb.set_trace()
            # if "vatex" in f:
            #     print("a")
            if f.endswith(".yaml"):
                yaml_path = os.path.join(root, f)
                try:
                    config = utils.load_yaml_config(yaml_path)

                    if "task" not in config:
                        continue

                    if register_task:
                        if type(config["task"]) == str:
                            register_configurable_task(config)
                    else:
                        if type(config["task"]) == list:
                            register_configurable_group(config)

                # Log this silently and show it only when
                # the user defines the appropriate verbosity.
                except ModuleNotFoundError as e:
                    eval_logger.debug(f"{yaml_path}: {e}. Config will not be added to registry.")
                except Exception as error:
                    import traceback

                    eval_logger.debug(f"Failed to load config in {yaml_path}. Config will not be added to registry\n" f"Error: {error}\n" f"Traceback: {traceback.format_exc()}")
    return 0


def include_path(task_dir):
    include_task_folder(task_dir)
    # Register Benchmarks after all tasks have been added
    include_task_folder(task_dir, register_task=False)
    return 0


def initialize_tasks(verbosity="INFO"):
    logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=verbosity)
    eval_logger.add(sys.stderr, level=verbosity)
    task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    include_path(task_dir)


def get_task(task_name, model_name):
    try:
        return TASK_REGISTRY[task_name](model_name=model_name)  # TODO choiszt the return result need to check " 'mmeConfigurableTask' object has no attribute '_instances'. Did you mean: 'instances'?"
    except KeyError:
        eval_logger.info("Available tasks:")
        eval_logger.info(list(TASK_REGISTRY) + list(GROUP_REGISTRY))
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return task_object.EVAL_HARNESS_NAME if hasattr(task_object, "EVAL_HARNESS_NAME") else type(task_object).__name__


# TODO: pass num_fewshot and other cmdline overrides in a better way
def get_task_dict(task_name_list: List[Union[str, Dict, Task]], model_name: str):
    all_task_dict = {}

    # Ensure task_name_list is a list to simplify processing
    if not isinstance(task_name_list, list):
        task_name_list = [task_name_list]

    for task_element in task_name_list:
        if isinstance(task_element, str) and task_element in GROUP_REGISTRY:
            group_name = task_element
            for task_name in GROUP_REGISTRY[task_element]:
                if task_name not in all_task_dict:
                    # Recursively get the task dictionary for nested groups
                    task_obj = get_task_dict([task_name], model_name)
                    # Merge the dictionaries
                    all_task_dict.update({task_name: (group_name, task_obj.get(task_name, None))})
        else:
            task_name = task_element if isinstance(task_element, str) else task_element.EVAL_HARNESS_NAME
            if task_name not in all_task_dict:
                task_obj = get_task(task_name=task_name, model_name=model_name)
                all_task_dict[task_name] = task_obj

    return all_task_dict

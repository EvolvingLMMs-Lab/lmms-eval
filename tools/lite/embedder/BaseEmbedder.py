import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch.distributed as dist

from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.tasks import (ConfigurableTask, get_task_dict, include_path,
                             initialize_tasks)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


class BaseEmbedder(ABC):
    def __init__(self, name: str, output_path: str) -> None:
        super().__init__()
        self.name = name
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        initialize_tasks()

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    # A static method to build requests for lmms_eval tasks
    # Pass in task name and return a list of Requests
    @staticmethod
    def init_task(task: str, ignored_ids: Union[set, List] = None):
        task_dict = get_task_dict([task], model_name="llava")
        task_obj = task_dict[task]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
        DATASET_PATH = task_obj.DATASET_PATH
        DATASET_NAME = None
        if task_obj.DATASET_NAME is not None:
            DATASET_NAME = task_obj.DATASET_NAME

        docs = task_obj.test_docs() if task_obj.has_test_docs() else task_obj.validation_docs()
        split = task_obj.config.test_split if task_obj.has_test_docs() else task_obj.config.validation_split
        rank0_print(f"\nTask : {task_obj.config.task}\n - #num : {len(task_obj.test_docs()) if task_obj.has_test_docs() else task_obj.validation_docs()}")
        task_obj.build_all_requests()
        requests = []
        for instance in task_obj.instances:
            reqtype = instance.request_type
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = instance.args
            if ignored_ids is not None and doc_id in ignored_ids:
                continue
            requests.append(instance)
        return DATASET_PATH, DATASET_NAME, split, requests, task_obj, docs

    @abstractmethod
    def embed_task(self, task: str):
        return

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from distutils.dir_util import copy_tree
from glob import glob
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.polynomial import polyfit
from sklearn.metrics import mean_squared_error

import lmms_eval
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.evaluator import evaluate
from lmms_eval.tasks import (
    ConfigurableTask,
    get_task_dict,
    include_path,
    initialize_tasks,
)
from lmms_eval.utils import simple_parse_args_string

eval_logger = logging.getLogger("lmms-eval")


class BaseShrinker(ABC):
    def __init__(self, task: str, num_items: Union[int, float], name: str, push_to_hub: bool = True) -> None:
        super().__init__()
        self.name = name
        self.task = task
        self.num_items = float(num_items)
        self.push_to_hub = push_to_hub

    @abstractmethod
    def shrink(self):
        return

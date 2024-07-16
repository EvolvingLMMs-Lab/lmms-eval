from abc import ABC, abstractmethod
from glob import glob
import os
import json
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Union
from distutils.dir_util import copy_tree
import random
import torch
import lmms_eval
from lmms_eval.evaluator import evaluate
from lmms_eval.tasks import initialize_tasks, include_path, get_task_dict, ConfigurableTask
from lmms_eval.api.registry import ALL_TASKS
import logging
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

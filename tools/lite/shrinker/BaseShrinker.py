import logging
from abc import ABC, abstractmethod
from typing import Union



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

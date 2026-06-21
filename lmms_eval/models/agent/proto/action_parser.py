from abc import ABC, abstractmethod


class ActionParser(ABC):
    """
    Convert the model's output into game-readable actions.
    """

    @abstractmethod
    def parse(self, action):
        pass

    def __call__(self, action):
        return self.parse(action)

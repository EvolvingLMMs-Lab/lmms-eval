from abc import ABC, abstractmethod


class ObservationParser(ABC):
    """
    Return the observation (the state of the game) to the model, then obtain content in a format the model can read.
    """

    @abstractmethod
    def parse(self, observation):
        pass

    def __call__(self, observation):
        return self.parse(observation)

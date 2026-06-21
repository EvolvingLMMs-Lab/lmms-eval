
from abc import ABC, abstractmethod


class AgentModel(ABC):
  @abstract
  def generate(self, ...):
    pass
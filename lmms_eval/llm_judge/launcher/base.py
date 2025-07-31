from abc import ABC, abstractmethod


class BaseLauncher(ABC):
    def __init__(self, port: int = 8000, host: str = "localhost", timeout: int = 1200, model: str = "Qwen/Qwen3-8B", **kwargs):
        super().__init__()
        self.port = port
        self.host = host
        self.timeout = timeout
        self.model = model

    @abstractmethod
    def launch(self, *args, **kwargs):
        """
        Launch the LLM judge with the given arguments.

        :param args: Positional arguments for the launch.
        :param kwargs: Keyword arguments for the launch.
        :return: The result of the launch operation.
        """
        pass

    @abstractmethod
    def clean():
        """
        Clean up resources or processes after the launch.

        :return: None
        """
        pass

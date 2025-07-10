from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

from .base import BaseLauncher


class SGLangLauncher(BaseLauncher):
    def __init__(self, port=8000, host="localhost", timeout=6000, model="Qwen/Qwen3-8B", mem_fraction_static: float = 0.83, tp: int = 8, api_key: str = None, **kwargs):
        super().__init__(port, host, timeout, model, **kwargs)
        self.mem_fraction_static = mem_fraction_static
        self.tp = tp
        self.base_url = f"http://{self.host}:{self.port}"
        self.api_key = api_key

    def launch(self, *args, **kwargs):
        """
        Launch the SGLang judge with the given arguments.

        :param args: Positional arguments for the launch.
        :param kwargs: Keyword arguments for the launch.
        :return: The result of the launch operation.
        """
        # Implement the logic to launch SGLang judge
        other_args = []
        other_args.extend(["--tensor-parallel-size", str(self.tp)])
        other_args.extend(["--mem-fraction-static", str(self.mem_fraction_static)])
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=self.timeout,
            api_key=self.api_key,
            other_args=other_args,
        )

    def clean(self):
        """
        Clean up resources or processes after the SGLang judge launch.

        :return: None
        """
        # Implement the cleanup logic for SGLang judge
        kill_process_tree(self.process.pid)

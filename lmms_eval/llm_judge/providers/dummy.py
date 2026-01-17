from typing import Optional

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class DummyProvider(ServerInterface):
    """OpenAI API implementation of the Judge interface"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)

    def is_available(self) -> bool:
        return True

    def evaluate(self, request: Request) -> Response:
        dummy_response = Response(content="dummy", model_used="dummy", usage="dummy", raw_response="dummy")
        return dummy_response

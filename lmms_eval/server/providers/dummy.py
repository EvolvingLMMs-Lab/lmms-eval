import base64
import os
import time
from io import BytesIO
from typing import Dict, List, Optional, Union

import requests
from loguru import logger as eval_logger
from PIL import Image

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class DummyProvider(ServerInterface):
    """OpenAI API implementation of the Judge interface"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)

    def is_available(self) -> bool:
        return True

    def evaluate(self, request: Request) -> Response:
        return "dummy"

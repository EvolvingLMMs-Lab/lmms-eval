"""MiniMax judge provider.

MiniMax exposes an OpenAI-compatible chat completions endpoint, so this
provider reuses :class:`OpenAIProvider` and only keeps the MiniMax-specific
defaults: credentials, the MiniMax-M3 model, temperature clamped to
``[0.0, 1.0]``, and ``<think>`` tag stripping for reasoning models.
"""

import os
import re
from dataclasses import replace
from typing import Any, Dict, Optional

from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig
from .openai import OpenAIProvider

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M3"
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _clamp_temperature(temperature: float) -> float:
    """Clamp temperature to MiniMax's accepted range [0.0, 1.0]."""
    return max(0.0, min(1.0, temperature))


def _strip_think_tags(text: str) -> str:
    """Strip ``<think>...</think>`` tags that MiniMax reasoning models may emit."""
    return _THINK_TAG_RE.sub("", text).strip()


def _default_config(config: Optional[ServerConfig]) -> ServerConfig:
    """Use MiniMax-M3 when the caller does not pass a config.

    ``ServerInterface`` otherwise fills in ``model_name="gpt-4"``.
    """
    if config is None:
        return ServerConfig(model_name=DEFAULT_MINIMAX_MODEL)
    return config


def _minimax_base_url() -> str:
    """Endpoint from MINIMAX_API_URL (for example the mainland endpoint or a proxy), else the global one."""
    return os.getenv("MINIMAX_API_URL", MINIMAX_BASE_URL).rstrip("/")


def _is_minimax_model(model_name: str) -> bool:
    """True for MiniMax model ids, including router-style ids such as ``minimax/minimax-m3``."""
    return model_name.lower().lstrip("/").split("/")[-1].startswith("minimax")


def _minimax_extra_body(config: ServerConfig) -> Dict[str, Any]:
    """Turn thinking off for MiniMax-M3.

    Thinking tokens count toward ``max_tokens``, so a short judge budget (for
    example 10) otherwise ends with ``finish_reason="length"`` and empty
    content. MiniMax-M3.1-Flash always thinks and returns 400 if thinking is
    disabled, so it is left alone and needs a larger ``max_tokens``.
    """
    if config.model_name.lower().split("/")[-1] == "minimax-m3":
        return {"thinking": {"type": "disabled"}}
    return {}


def _request_for_minimax(request: Request, fallback: ServerConfig) -> Request:
    """Return a request whose config is valid for MiniMax.

    The caller's config object is left unchanged. A copy is attached only
    when something has to change: the temperature is clamped, and a model
    name that is not a MiniMax model (tasks default to a GPT name such as
    ``gpt-4o-2024-11-20``) is replaced with ``MINIMAX_MODEL`` or MiniMax-M3.
    """
    config = request.config or fallback
    changes: Dict[str, Any] = {}
    clamped = _clamp_temperature(config.temperature)
    if clamped != config.temperature:
        changes["temperature"] = clamped
    if not _is_minimax_model(config.model_name):
        model = os.getenv("MINIMAX_MODEL") or DEFAULT_MINIMAX_MODEL
        _warn_model_replaced(config.model_name, model)
        changes["model_name"] = model
    if not changes:
        return request
    return replace(request, config=replace(config, **changes))


_warned_models = set()


def _warn_model_replaced(requested: str, used: str) -> None:
    """Log once per requested name that a non-MiniMax model name was replaced."""
    if requested not in _warned_models:
        _warned_models.add(requested)
        eval_logger.warning(f"MiniMax judge: '{requested}' is not a MiniMax model; using '{used}'. Set MINIMAX_MODEL to choose another.")


class MiniMaxProvider(OpenAIProvider):
    """MiniMax API implementation of the Judge interface.

    Supported models include ``MiniMax-M3`` (default; 512K context, 128K
    max output, image input supported), ``MiniMax-M2.7``, and
    ``MiniMax-M2.7-highspeed``.

    Environment variables
    ---------------------
    MINIMAX_API_KEY : str
        API key for the MiniMax platform. ``API_TYPE=minimax`` selects this
        provider with no other configuration.
    MINIMAX_MODEL : str, optional
        Model used when a task asks for a non-MiniMax name (its GPT default).
        Defaults to ``MiniMax-M3``.
    MINIMAX_API_URL : str, optional
        Base URL, for example the mainland endpoint. Defaults to
        ``https://api.minimax.io/v1``.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        # Skip OpenAIProvider.__init__: it builds a client from OPENAI_API_KEY,
        # which the documented MiniMax setup (`API_TYPE=minimax` plus
        # MINIMAX_API_KEY) does not set. Request handling still comes from
        # OpenAIProvider.evaluate.
        ServerInterface.__init__(self, _default_config(config))
        self.api_key = os.getenv("MINIMAX_API_KEY", "")
        self.base_url = _minimax_base_url()
        self.client = None
        self.use_client = False
        if not self.api_key:
            return
        # Deferred so a missing openai package still falls back to requests,
        # matching OpenAIProvider.
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_client = True
        except ImportError:
            eval_logger.warning("OpenAI client not available, falling back to requests for MiniMax")

    def _extra_body(self, config: ServerConfig) -> Dict[str, Any]:
        return _minimax_extra_body(config)

    def evaluate(self, request: Request) -> Response:
        """Evaluate using the MiniMax API."""
        if not self.is_available():
            raise ValueError("MiniMax API key not configured (set MINIMAX_API_KEY)")
        response = super().evaluate(_request_for_minimax(request, self.config))
        response.content = _strip_think_tags(response.content)
        return response

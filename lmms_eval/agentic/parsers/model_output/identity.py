from __future__ import annotations

from typing import Any

from lmms_eval.agentic.parsers.base import ModelOutputParser, ParserContext
from lmms_eval.agentic.registry_core import register_model_output_parser


@register_model_output_parser("identity", replace=True)
class IdentityModelOutputParser(ModelOutputParser):
    """Pass model output through unchanged."""

    def parse(self, value: Any, ctx: ParserContext) -> Any:
        del ctx
        return value

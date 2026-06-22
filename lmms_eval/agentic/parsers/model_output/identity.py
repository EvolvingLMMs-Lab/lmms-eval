from __future__ import annotations

from typing import Any

from lmms_eval.agentic.parsers.base import ModelOutputParser, ParserContext


class IdentityModelOutputParser(ModelOutputParser):
    """Pass model output through unchanged."""

    def parse(self, value: Any, ctx: ParserContext) -> Any:
        del ctx
        return value

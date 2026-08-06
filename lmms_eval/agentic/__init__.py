"""Agentic game-loop evaluation (`output_type: generate_until_game`).

This package is the single public surface. Everything imported here is
lightweight; heavy or optional dependencies (openai, av, vizdoom) load only
when the matching component is actually built.
"""

from lmms_eval.agentic.components import (
    REGISTRY,
    call_with_accepted_kwargs,
    import_from_path,
    resolve,
)
from lmms_eval.agentic.env import EnvManager
from lmms_eval.agentic.episode import run_episode
from lmms_eval.agentic.pipelines import apply_parser_pipeline, select_parser_pipelines
from lmms_eval.agentic.runner import run_generate_until_game
from lmms_eval.agentic.servers import (
    FixedActionModelServer,
    ModelServer,
    OpenAIModelServer,
)
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    EpisodeResult,
    EpisodeStep,
    GameAction,
    ParsedAction,
    Parser,
    ParserContext,
    StepResult,
)

__all__ = [
    "REGISTRY",
    "AgentInput",
    "AgentOutput",
    "ContentBlock",
    "EnvManager",
    "EnvState",
    "EpisodeResult",
    "EpisodeStep",
    "FixedActionModelServer",
    "GameAction",
    "ModelServer",
    "OpenAIModelServer",
    "Parser",
    "ParsedAction",
    "ParserContext",
    "StepResult",
    "apply_parser_pipeline",
    "call_with_accepted_kwargs",
    "import_from_path",
    "resolve",
    "run_episode",
    "run_generate_until_game",
    "select_parser_pipelines",
]

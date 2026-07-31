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
from lmms_eval.agentic.parsers import (
    ActionNameParser,
    ActionParser,
    IdentityModelOutputParser,
    ModelOutputParser,
    ObservationParser,
    ParserContext,
    QwenModelOutputParser,
)
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
    StepResult,
)

__all__ = [
    "REGISTRY",
    "ActionNameParser",
    "ActionParser",
    "AgentInput",
    "AgentOutput",
    "ContentBlock",
    "EnvManager",
    "EnvState",
    "EpisodeResult",
    "EpisodeStep",
    "FixedActionModelServer",
    "GameAction",
    "IdentityModelOutputParser",
    "ModelOutputParser",
    "ModelServer",
    "ObservationParser",
    "OpenAIModelServer",
    "ParsedAction",
    "ParserContext",
    "QwenModelOutputParser",
    "StepResult",
    "call_with_accepted_kwargs",
    "import_from_path",
    "resolve",
    "run_episode",
    "run_generate_until_game",
]

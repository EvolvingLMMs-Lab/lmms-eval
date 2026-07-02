from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RolloutEpisodeSpec:
    """Resolved lmms-eval episode configuration for rollout workers.

    This mirrors generate_until_game task config, but it is already resolved to
    a concrete doc plus component specs. VizDoom task YAML is the reference shape.
    """

    doc: Any
    game_env: Any
    observation_parser: Any
    action_parser: Any
    model_server: Any = "openai"
    loop_worker: Any = "simple"
    model_output_parser: Any = "identity"
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    lmms_eval_specific_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 32
    seed: int | None = None
    agent_id: str = "agent"
    request_metadata: dict[str, Any] = field(default_factory=dict)

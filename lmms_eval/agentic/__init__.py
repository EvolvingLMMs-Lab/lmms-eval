"""Minimal agentic/game evaluation interfaces."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ActionNameParser": "lmms_eval.agentic.parsers",
    "ActionParser": "lmms_eval.agentic.parsers",
    "AgenticFactory": "lmms_eval.agentic.factory",
    "AgentInput": "lmms_eval.agentic.types",
    "AgentModel": "lmms_eval.agentic.model_server",
    "AgentOutput": "lmms_eval.agentic.types",
    "ContentBlock": "lmms_eval.agentic.types",
    "DEFAULT_AGENTIC_FACTORY": "lmms_eval.agentic.factory",
    "EnvManager": "lmms_eval.agentic.env",
    "EnvState": "lmms_eval.agentic.types",
    "EpisodeResult": "lmms_eval.agentic.types",
    "EpisodeStep": "lmms_eval.agentic.types",
    "GameAction": "lmms_eval.agentic.types",
    "IdentityModelOutputParser": "lmms_eval.agentic.parsers",
    "LoopManager": "lmms_eval.agentic.loop",
    "LoopSession": "lmms_eval.agentic.loop",
    "LoopWorker": "lmms_eval.agentic.loop",
    "ModelOutputParser": "lmms_eval.agentic.parsers",
    "ModelServer": "lmms_eval.agentic.model_server",
    "ObservationParser": "lmms_eval.agentic.parsers",
    "OpenAIModelServer": "lmms_eval.agentic.model_server",
    "ParsedAction": "lmms_eval.agentic.types",
    "Parser": "lmms_eval.agentic.parsers",
    "ParserContext": "lmms_eval.agentic.parsers",
    "QwenModelOutputParser": "lmms_eval.agentic.parsers",
    "RolloutJob": "lmms_eval.agentic.loop",
    "SimpleLoopSession": "lmms_eval.agentic.loop",
    "SimpleLoopWorker": "lmms_eval.agentic.loop",
    "SingleAgentLoopWorker": "lmms_eval.agentic.loop",
    "StepResult": "lmms_eval.agentic.types",
    "VizDoomActionParser": "lmms_eval.agentic.parsers",
    "VizDoomObservationParser": "lmms_eval.agentic.parsers",
    "build_action_parser": "lmms_eval.agentic.factory",
    "build_env_manager": "lmms_eval.agentic.factory",
    "build_loop_worker": "lmms_eval.agentic.factory",
    "build_model_output_parser": "lmms_eval.agentic.factory",
    "build_model_server": "lmms_eval.agentic.factory",
    "build_observation_parser": "lmms_eval.agentic.factory",
    "run_generate_until_game": "lmms_eval.agentic.loop",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

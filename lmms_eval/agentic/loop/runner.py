from __future__ import annotations

import json
from typing import Any

from lmms_eval.agentic.registry import build_action_parser, build_game_env, build_loop_worker, build_model_output_parser, build_model_server, build_observation_parser
from lmms_eval.agentic.types import EpisodeResult, GameAction
from lmms_eval.api.instance import Instance


def run_generate_until_game(lm: Any, requests: list[Instance], response_cache: Any = None) -> list[str]:
    outputs = []
    for req in requests:
        if len(req.args) == 10:
            ctx, generation_kwargs, _doc_to_visual, game_env, observation_parser, action_parser, lmms_eval_specific_kwargs, doc_id, task_name, split = req.args
            model_server_spec = "lmms"
            loop_worker_spec = "simple"
            model_output_parser_spec = "identity"
        elif len(req.args) == 12:
            (
                ctx,
                generation_kwargs,
                _doc_to_visual,
                model_server_spec,
                loop_worker_spec,
                game_env,
                observation_parser,
                action_parser,
                lmms_eval_specific_kwargs,
                doc_id,
                task_name,
                split,
            ) = req.args
            model_output_parser_spec = "identity"
        else:
            (
                ctx,
                generation_kwargs,
                _doc_to_visual,
                model_server_spec,
                loop_worker_spec,
                game_env,
                observation_parser,
                model_output_parser_spec,
                action_parser,
                lmms_eval_specific_kwargs,
                doc_id,
                task_name,
                split,
            ) = req.args
        del ctx

        doc = lm.task_dict[task_name][split][doc_id]
        gen_kwargs = dict(generation_kwargs or {})
        max_steps = int(gen_kwargs.pop("max_game_steps", gen_kwargs.pop("max_agentic_steps", 32)))
        seed = gen_kwargs.pop("game_seed", None)

        model_server = build_model_server(
            model_server_spec,
            lm=lm,
            generation_kwargs=gen_kwargs,
            doc_id=doc_id,
            task_name=task_name,
            split=split,
            request_metadata=req.metadata,
            response_cache=response_cache,
        )
        worker = build_loop_worker(
            loop_worker_spec,
            model_server=model_server,
            env=build_game_env(game_env, doc=doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs),
            observation_parser=build_observation_parser(observation_parser, doc=doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs),
            model_output_parser=build_model_output_parser(model_output_parser_spec, doc=doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs),
            action_parser=build_action_parser(action_parser, doc=doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs),
            max_steps=max_steps,
        )
        outputs.append(_episode_to_json(worker.run(doc, seed=seed)))
    return outputs


def _episode_to_json(result: EpisodeResult) -> str:
    payload = {
        "success": result.success,
        "metrics": result.metrics,
        "final_state": {
            "env_id": result.final_state.env_id,
            "step_idx": result.final_state.step_idx,
            "observation": result.final_state.observation,
            "terminal": result.final_state.terminal,
            "metadata": result.final_state.metadata,
        },
        "steps": [
            {
                "step_idx": step.state.step_idx,
                "raw_model_output": step.raw_output.first_text() if step.raw_output is not None else None,
                "model_output": step.output.first_text() if step.output is not None else None,
                "action": _action_to_dict(step.parsed_action.action) if step.parsed_action is not None else None,
                "parse_error": step.parsed_action.error if step.parsed_action is not None else None,
                "reward": step.result.reward if step.result is not None else None,
                "done": step.result.done if step.result is not None else None,
                "info": step.result.info if step.result is not None else {},
            }
            for step in result.steps
        ],
        "metadata": result.metadata,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _action_to_dict(action: GameAction | dict[str, GameAction] | None):
    if action is None:
        return None
    if isinstance(action, dict):
        return {agent_id: _action_to_dict(agent_action) for agent_id, agent_action in action.items()}
    return {"type": action.type, "data": action.data, "agent_id": action.agent_id, "metadata": action.metadata}

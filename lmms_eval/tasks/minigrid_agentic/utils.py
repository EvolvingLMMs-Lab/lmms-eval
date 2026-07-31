"""MiniGrid task glue: docs in, metrics out.

The environment side lives entirely behind ``game_env: minigrid`` in the YAML
(registry env + default parsers), so this module only defines the scenario
list and maps episode JSON to task metrics. Scenarios are docs-from-code
(``dataset_path: !function utils.minigrid_dataset``): for env-loop tasks the
"dataset" is a handful of (env_id, seed, instruction) configs, which belong in
code, not in a tracked data file or a hub dataset.
"""

import json

_SCENARIOS = [
    {"env_id": "MiniGrid-Empty-6x6-v0", "seed": 1, "instruction": "You control the red agent in a grid world. Reach the green goal square."},
    {"env_id": "MiniGrid-Empty-8x8-v0", "seed": 7, "instruction": "You control the red agent in a grid world. Reach the green goal square."},
    {"env_id": "MiniGrid-DoorKey-6x6-v0", "seed": 3, "instruction": "You control the red agent in a grid world. Pick up the key, open the locked door, then reach the green goal square."},
]


def minigrid_dataset():
    import datasets

    return datasets.DatasetDict({"test": datasets.Dataset.from_list(_SCENARIOS)})


def minigrid_doc_to_visual(doc):
    return []


def minigrid_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{doc.get('instruction', '')}{kwargs.get('post_prompt', '')}"


def minigrid_doc_to_target(doc):
    return "reach_goal"


def minigrid_process_results(doc, results):
    raw = results[0] if results else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"minigrid_success": 0.0, "minigrid_steps": 0.0, "minigrid_invalid_actions": 0.0, "minigrid_env_error": 1.0}

    metrics = payload.get("metrics", {})
    return {
        "minigrid_success": 1.0 if payload.get("success") else 0.0,
        "minigrid_steps": float(metrics.get("steps", 0.0)),
        "minigrid_invalid_actions": float(metrics.get("invalid_actions", 0.0)),
        "minigrid_env_error": float(metrics.get("env_error", 0.0)),
    }


def minigrid_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0

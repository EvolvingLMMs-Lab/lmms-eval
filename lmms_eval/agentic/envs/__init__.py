"""Reusable environments for the agentic game loop.

Each module wraps one environment family and is referenced by registry name
from ``components.REGISTRY["env_manager"]`` (e.g. ``game_env: minigrid`` in a
task YAML). Modules must keep heavy dependencies out of import time: import
the simulator inside ``reset`` so building the manager never requires it.

Bespoke, single-task environments stay next to their task under
``lmms_eval/tasks/<task>/env.py`` instead of here.
"""

"""Unified pipeline tests for mainstream evaluation tasks.

Validates that representative tasks are correctly registered, their YAML
configs load without error, and their utils functions (doc_to_text,
process_results, aggregation) are importable and callable.

No dataset download, no model inference, no API keys required.
"""

import importlib
import os

import pytest
import yaml

from lmms_eval.tasks import TaskManager

# ---------------------------------------------------------------------------
# Shared TaskManager (expensive to init, reuse across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tm():
    """Module-scoped fixture for TaskManager singleton."""
    return TaskManager("WARNING")


# ---------------------------------------------------------------------------
# Mainstream tasks to validate
# ---------------------------------------------------------------------------

MAINSTREAM_TASKS = [
    "mme",
    "mmmu_val",
    "mmstar",
    "ai2d",
    "scienceqa",
    "ocrbench",
    "mmvet",
    "videomme",
]

TASKS_WITH_DOC_TO_MESSAGES = {"mmmu_val"}

MAINSTREAM_GROUPS = [
    "mme",
    "mmmu",
    "mmbench",
    "mathvista",
]

TASK_UTILS = {
    "mme": "lmms_eval.tasks.mme.utils",
    "mmmu_val": "lmms_eval.tasks.mmmu.utils",
    "mmstar": "lmms_eval.tasks.mmstar.utils",
    "ai2d": "lmms_eval.tasks.ai2d.utils",
    "scienceqa": "lmms_eval.tasks.scienceqa.utils",
    "ocrbench": "lmms_eval.tasks.ocrbench.utils",
    "mmvet": "lmms_eval.tasks.mmvet.utils",
    "videomme": "lmms_eval.tasks.videomme.utils",
}

TASK_PROCESS_RESULTS_FN = {
    "mme": "mme_process_results",
    "mmmu_val": "mmmu_process_results",
    "mmstar": "mmstar_process_results",
    "scienceqa": "sqa_process_results",
    "ocrbench": "ocrbench_process_results",
    "mmvet": "mmvet_process_results",
    "videomme": "videomme_process_results",
}


# ===========================================================================
# YAML Config Integrity Helpers
# ===========================================================================


class _FunctionTag:
    """Placeholder for !function YAML tags during safe loading."""

    def __init__(self, value):
        self.value = value


def _function_constructor(loader, node):
    return _FunctionTag(loader.construct_scalar(node))


def _load_task_yaml(yaml_path: str) -> dict:
    loader = yaml.SafeLoader
    loader_copy = type("SafeLoaderCopy", (loader,), {})
    loader_copy.add_constructor("!function", _function_constructor)
    with open(yaml_path) as f:
        return yaml.load(f, Loader=loader_copy)


# ===========================================================================
# Task Registration
# ===========================================================================


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_mainstream_tasks_are_registered(tm, task_name):
    """Verify each mainstream task is registered in TaskManager."""
    assert task_name in tm.all_subtasks, f"Task '{task_name}' not found in TaskManager.all_subtasks"


@pytest.mark.parametrize("group_name", MAINSTREAM_GROUPS)
def test_mainstream_groups_are_registered(tm, group_name):
    """Verify each mainstream group is registered in TaskManager."""
    assert group_name in tm.all_tasks, f"Group '{group_name}' not found in TaskManager.all_tasks"


def test_task_count_is_reasonable(tm):
    """Verify the registry contains a reasonable number of subtasks."""
    assert len(tm.all_subtasks) > 100, "Expected 100+ subtasks in the registry"


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_task_index_has_yaml_paths(tm, task_name):
    """Verify each task has a valid YAML path in the index."""
    entry = tm.task_index.get(task_name)
    assert entry is not None, f"No index entry for '{task_name}'"
    yaml_path = entry.get("yaml_path")
    assert yaml_path is not None, f"No yaml_path for '{task_name}'"
    assert os.path.isfile(yaml_path), f"YAML file missing: {yaml_path}"


# ===========================================================================
# YAML Config Integrity
# ===========================================================================


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_yaml_files_parse_without_error(tm, task_name):
    """Verify YAML files parse to valid dicts."""
    yaml_path = tm.task_index[task_name]["yaml_path"]
    cfg = _load_task_yaml(yaml_path)
    assert isinstance(cfg, dict), f"YAML for '{task_name}' did not parse to a dict"


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_yaml_has_task_field(tm, task_name):
    """Verify each YAML has a 'task' field."""
    yaml_path = tm.task_index[task_name]["yaml_path"]
    cfg = _load_task_yaml(yaml_path)
    task_field = cfg.get("task")
    assert task_field is not None, f"YAML for '{task_name}' missing 'task' field"


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_yaml_has_dataset_path(tm, task_name):
    """Verify each YAML has either 'dataset_path' or 'include'."""
    yaml_path = tm.task_index[task_name]["yaml_path"]
    cfg = _load_task_yaml(yaml_path)
    has_dataset = "dataset_path" in cfg or "include" in cfg
    assert has_dataset, f"YAML for '{task_name}' has neither 'dataset_path' nor 'include'"


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_yaml_has_formatter(tm, task_name):
    """Verify each YAML has an input formatter."""
    yaml_path = tm.task_index[task_name]["yaml_path"]
    cfg = _load_task_yaml(yaml_path)
    has_formatter = any(k in cfg or "include" in cfg for k in ("doc_to_messages", "doc_to_text", "doc_to_visual"))
    assert has_formatter, f"YAML for '{task_name}' has no input formatter and no include"


# ===========================================================================
# Utils Module & Functions
# ===========================================================================


@pytest.mark.parametrize("task_name,module_path", TASK_UTILS.items())
def test_utils_modules_are_importable(task_name, module_path):
    """Verify utils modules can be imported."""
    mod = importlib.import_module(module_path)
    assert mod is not None


@pytest.mark.parametrize("task_name,fn_name", TASK_PROCESS_RESULTS_FN.items())
def test_process_results_functions_exist_and_callable(task_name, fn_name):
    """Verify process_results functions exist and are callable."""
    module_path = TASK_UTILS[task_name]
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name, None)
    assert fn is not None, f"{module_path}.{fn_name} does not exist"
    assert callable(fn), f"{module_path}.{fn_name} is not callable"


def test_doc_to_text_functions_exist():
    """Verify doc_to_text functions exist for known tasks."""
    known_doc_to_text = {
        "mme": "mme_doc_to_text",
        "mmmu_val": "mmmu_doc_to_text",
        "mmstar": "mmstar_doc_to_text",
        "scienceqa": "sqa_doc_to_text",
        "ocrbench": "ocrbench_doc_to_text",
        "videomme": "videomme_doc_to_text",
    }
    for task_name, fn_name in known_doc_to_text.items():
        module_path = TASK_UTILS[task_name]
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name, None)
        assert fn is not None, f"{module_path}.{fn_name} does not exist"
        assert callable(fn), f"{module_path}.{fn_name} is not callable"


def test_doc_to_messages_exists_for_chat_tasks():
    """Verify doc_to_messages exists for tasks that use it."""
    for task_name in TASKS_WITH_DOC_TO_MESSAGES:
        module_path = TASK_UTILS[task_name]
        mod = importlib.import_module(module_path)
        fn = getattr(mod, "mmmu_doc_to_messages", None)
        assert fn is not None, f"{module_path} missing doc_to_messages function"
        assert callable(fn)


# ===========================================================================
# Cross-task consistency
# ===========================================================================


@pytest.mark.parametrize("task_name", MAINSTREAM_TASKS)
def test_all_mainstream_tasks_have_generate_until_output_type(tm, task_name):
    """Verify mainstream tasks use generate_until output type."""
    yaml_path = tm.task_index[task_name]["yaml_path"]
    cfg = _load_task_yaml(yaml_path)
    output_type = cfg.get("output_type")
    if output_type is not None:
        assert output_type == "generate_until", f"Task '{task_name}' has unexpected output_type: {output_type}"


def test_no_duplicate_task_names_in_registry(tm):
    """Verify no duplicate task names exist in the registry."""
    seen = set()
    for name in tm.all_subtasks:
        assert name not in seen, f"Duplicate subtask name: {name}"
        seen.add(name)

"""Unified pipeline tests for mainstream evaluation tasks.

Validates that representative tasks are correctly registered, their YAML
configs load without error, and their utils functions (doc_to_text,
process_results, aggregation) are importable and callable.

No dataset download, no model inference, no API keys required.
"""

import importlib
import os
import unittest

import yaml

from lmms_eval.tasks import TaskManager

# ---------------------------------------------------------------------------
# Shared TaskManager (expensive to init, reuse across all tests)
# ---------------------------------------------------------------------------

_tm: TaskManager = None


def _get_tm() -> TaskManager:
    global _tm
    if _tm is None:
        _tm = TaskManager("WARNING")
    return _tm


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
# Task Registration
# ===========================================================================


class TestTaskRegistration(unittest.TestCase):
    def test_mainstream_tasks_are_registered(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                self.assertIn(task_name, tm.all_subtasks, f"Task '{task_name}' not found in TaskManager.all_subtasks")

    def test_mainstream_groups_are_registered(self):
        tm = _get_tm()
        for group_name in MAINSTREAM_GROUPS:
            with self.subTest(group=group_name):
                self.assertIn(group_name, tm.all_tasks, f"Group '{group_name}' not found in TaskManager.all_tasks")

    def test_task_count_is_reasonable(self):
        tm = _get_tm()
        self.assertGreater(len(tm.all_subtasks), 100, "Expected 100+ subtasks in the registry")

    def test_task_index_has_yaml_paths(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                entry = tm.task_index.get(task_name)
                self.assertIsNotNone(entry, f"No index entry for '{task_name}'")
                yaml_path = entry.get("yaml_path")
                self.assertIsNotNone(yaml_path, f"No yaml_path for '{task_name}'")
                self.assertTrue(os.path.isfile(yaml_path), f"YAML file missing: {yaml_path}")


# ===========================================================================
# YAML Config Integrity
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


class TestYAMLConfigIntegrity(unittest.TestCase):
    def test_yaml_files_parse_without_error(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                yaml_path = tm.task_index[task_name]["yaml_path"]
                cfg = _load_task_yaml(yaml_path)
                self.assertIsInstance(cfg, dict, f"YAML for '{task_name}' did not parse to a dict")

    def test_yaml_has_task_field(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                yaml_path = tm.task_index[task_name]["yaml_path"]
                cfg = _load_task_yaml(yaml_path)
                task_field = cfg.get("task")
                self.assertIsNotNone(task_field, f"YAML for '{task_name}' missing 'task' field")

    def test_yaml_has_dataset_path(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                yaml_path = tm.task_index[task_name]["yaml_path"]
                cfg = _load_task_yaml(yaml_path)
                has_dataset = "dataset_path" in cfg or "include" in cfg
                self.assertTrue(has_dataset, f"YAML for '{task_name}' has neither 'dataset_path' nor 'include'")

    def test_yaml_has_formatter(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                yaml_path = tm.task_index[task_name]["yaml_path"]
                cfg = _load_task_yaml(yaml_path)
                has_formatter = any(k in cfg or "include" in cfg for k in ("doc_to_messages", "doc_to_text", "doc_to_visual"))
                self.assertTrue(has_formatter, f"YAML for '{task_name}' has no input formatter and no include")


# ===========================================================================
# Utils Module & Functions
# ===========================================================================


class TestUtilsFunctions(unittest.TestCase):
    def test_utils_modules_are_importable(self):
        for task_name, module_path in TASK_UTILS.items():
            with self.subTest(task=task_name):
                mod = importlib.import_module(module_path)
                self.assertIsNotNone(mod)

    def test_process_results_functions_exist_and_callable(self):
        for task_name, fn_name in TASK_PROCESS_RESULTS_FN.items():
            with self.subTest(task=task_name):
                module_path = TASK_UTILS[task_name]
                mod = importlib.import_module(module_path)
                fn = getattr(mod, fn_name, None)
                self.assertIsNotNone(fn, f"{module_path}.{fn_name} does not exist")
                self.assertTrue(callable(fn), f"{module_path}.{fn_name} is not callable")

    def test_doc_to_text_functions_exist(self):
        known_doc_to_text = {
            "mme": "mme_doc_to_text",
            "mmmu_val": "mmmu_doc_to_text",
            "mmstar": "mmstar_doc_to_text",
            "scienceqa": "sqa_doc_to_text",
            "ocrbench": "ocrbench_doc_to_text",
            "videomme": "videomme_doc_to_text",
        }
        for task_name, fn_name in known_doc_to_text.items():
            with self.subTest(task=task_name):
                module_path = TASK_UTILS[task_name]
                mod = importlib.import_module(module_path)
                fn = getattr(mod, fn_name, None)
                self.assertIsNotNone(fn, f"{module_path}.{fn_name} does not exist")
                self.assertTrue(callable(fn), f"{module_path}.{fn_name} is not callable")

    def test_doc_to_messages_exists_for_chat_tasks(self):
        for task_name in TASKS_WITH_DOC_TO_MESSAGES:
            with self.subTest(task=task_name):
                module_path = TASK_UTILS[task_name]
                mod = importlib.import_module(module_path)
                fn = getattr(mod, "mmmu_doc_to_messages", None)
                self.assertIsNotNone(fn, f"{module_path} missing doc_to_messages function")
                self.assertTrue(callable(fn))


# ===========================================================================
# Cross-task consistency
# ===========================================================================


class TestCrossTaskConsistency(unittest.TestCase):
    def test_all_mainstream_tasks_have_generate_until_output_type(self):
        tm = _get_tm()
        for task_name in MAINSTREAM_TASKS:
            with self.subTest(task=task_name):
                yaml_path = tm.task_index[task_name]["yaml_path"]
                cfg = _load_task_yaml(yaml_path)
                output_type = cfg.get("output_type")
                if output_type is not None:
                    self.assertEqual(
                        output_type,
                        "generate_until",
                        f"Task '{task_name}' has unexpected output_type: {output_type}",
                    )

    def test_no_duplicate_task_names_in_registry(self):
        tm = _get_tm()
        seen = set()
        for name in tm.all_subtasks:
            self.assertNotIn(name, seen, f"Duplicate subtask name: {name}")
            seen.add(name)


if __name__ == "__main__":
    unittest.main()

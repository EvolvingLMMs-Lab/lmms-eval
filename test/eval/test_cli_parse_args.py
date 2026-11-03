import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from lmms_eval import __main__ as main_module


class TestCliParseArgs(unittest.TestCase):
    def test_max_tokens_arg_is_parsed(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--max_tokens",
            "12345",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertEqual(args.max_tokens, 12345)

    def test_max_tokens_defaults_to_none(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertIsNone(args.max_tokens)

    def test_process_docs_parallel_defaults_to_four(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertEqual(args.process_docs_parallel, 4)

    def test_process_docs_parallel_accepts_hyphenated_option(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--process-docs-parallel",
            "7",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertEqual(args.process_docs_parallel, 7)

    def test_process_docs_parallel_rejects_zero(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--process-docs-parallel",
            "0",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            main_module.parse_eval_args()


class TestCliTaskFiles(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.first = self.directory / "first.yaml"
        self.second = self.directory / "second.yaml"
        self.first.write_text("task: first\noutput_type: generate_until\n")
        self.second.write_text("task: second\noutput_type: generate_until\n")

    def _evaluate_tasks(self, tasks: str, matched_tasks: tuple[str, ...] = ()) -> list:
        argv = ["lmms-eval", "--model", "dummy", "--tasks", tasks]
        with (
            patch.object(sys, "argv", argv),
            patch.object(main_module, "TaskManager") as manager,
            patch.object(main_module, "EvaluationTracker"),
            patch.object(main_module.evaluator, "simple_evaluate", return_value=None) as evaluate,
        ):
            manager.return_value.match_tasks.return_value = list(matched_tasks)
            _, args = main_module.parse_eval_args()
            main_module.cli_evaluate_single(args)
        return evaluate.call_args.kwargs["tasks"]

    def test_single_yaml_file_is_passed_to_evaluator(self) -> None:
        tasks = self._evaluate_tasks(str(self.first))

        self.assertEqual(tasks, [{"task": "first", "output_type": "generate_until"}])

    def test_multiple_yaml_files_are_passed_to_evaluator(self) -> None:
        tasks = self._evaluate_tasks(f"{self.first},{self.second}")

        self.assertEqual([task["task"] for task in tasks], ["first", "second"])

    def test_yaml_file_can_be_mixed_with_registered_tasks_and_wildcards(self) -> None:
        tasks = self._evaluate_tasks(f"mme,mmmu_*,{self.first}", ("mme", "mmmu_val"))

        self.assertEqual(tasks[:2], ["mme", "mmmu_val"])
        self.assertEqual(tasks[2], {"task": "first", "output_type": "generate_until"})

    def test_missing_file_is_still_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Tasks not found: .*missing.yaml"):
            self._evaluate_tasks(str(self.directory / "missing.yaml"))

    def test_mixed_file_and_unknown_task_only_reports_unknown_task(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Tasks not found: missing_task\. Try"):
            self._evaluate_tasks(f"{self.first},missing_task")

    def test_directory_still_loads_all_yaml_files(self) -> None:
        tasks = self._evaluate_tasks(str(self.directory))

        self.assertEqual({task["task"] for task in tasks}, {"first", "second"})

    def test_registered_tasks_and_wildcards_are_unchanged(self) -> None:
        self.assertEqual(self._evaluate_tasks("mme,mmmu_*", ("mme", "mmmu_val")), ["mme", "mmmu_val"])

    def _assert_local_task_request_name(self, task_type: Literal["simple", "chat"]) -> None:
        data_file = self.directory / "data.jsonl"
        data_file.write_text('{"question": "What color?", "answer": "red"}\n')
        config = {
            "task": "unregistered_local_task",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(data_file)}},
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "process_results": lambda doc, results: {"accuracy": float(results[0] == doc["answer"])},
            "metric_list": [{"metric": "accuracy", "aggregation": "mean", "higher_is_better": True}],
        }
        task = main_module.TaskManager(include_defaults=False).load_config(config, task_type=task_type)["unregistered_local_task"]
        task.build_all_requests(limit=1, rank=0, world_size=1)

        self.assertEqual(task.config.task, "unregistered_local_task")
        self.assertEqual(task.instances[0].args[-2], "unregistered_local_task")

    def test_unregistered_simple_task_keeps_its_request_name(self) -> None:
        self._assert_local_task_request_name("simple")

    def test_unregistered_chat_task_keeps_its_request_name(self) -> None:
        self._assert_local_task_request_name("chat")


if __name__ == "__main__":
    unittest.main()

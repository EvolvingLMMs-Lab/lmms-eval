import unittest

from lmms_eval.tasks import TaskManager


class TestBenchmarkAliasRegistration(unittest.TestCase):
    def test_alias_groups_are_registered(self):
        task_manager = TaskManager()
        expected_groups = {"anet_qa", "mmmu_a", "egosch_a"}
        missing_groups = expected_groups.difference(task_manager.all_tasks)
        self.assertFalse(missing_groups, f"Missing benchmark alias groups: {sorted(missing_groups)}")

    def test_alias_group_yaml_targets(self):
        task_manager = TaskManager()
        expected_yaml_targets = {
            "anet_qa": "activitynetqa",
            "mmmu_a": "mmmu_val",
            "egosch_a": "egoschema",
        }
        for group_name, expected_task in expected_yaml_targets.items():
            yaml_path = task_manager.task_index[group_name]["yaml_path"]
            with open(yaml_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            self.assertIn(f"  - {expected_task}", content)


if __name__ == "__main__":
    unittest.main()

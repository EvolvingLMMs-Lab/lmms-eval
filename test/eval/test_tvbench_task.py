import unittest
from unittest.mock import patch

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.tvbench import utils


class TestTVBenchTaskRegistration(unittest.TestCase):
    def test_tvbench_group_and_subtasks_are_registered(self):
        task_manager = TaskManager()
        expected_subtasks = {
            "tvbench_action_antonym",
            "tvbench_action_count",
            "tvbench_action_localization",
            "tvbench_action_sequence",
            "tvbench_egocentric_sequence",
            "tvbench_moving_direction",
            "tvbench_object_count",
            "tvbench_object_shuffle",
            "tvbench_scene_transition",
            "tvbench_unexpected_action",
        }

        self.assertIn("tvbench", task_manager.all_groups)
        available_tvbench_subtasks = {task for task in task_manager.all_subtasks if task.startswith("tvbench_")}
        self.assertSetEqual(available_tvbench_subtasks, expected_subtasks)


class TestTVBenchUtils(unittest.TestCase):
    def setUp(self):
        self.doc = {
            "question": "What is the person doing?",
            "candidates": ["Running", "Sitting", "Jumping", "Standing"],
            "answer": "Sitting",
            "video": "sample_video.mp4",
        }

    def test_doc_to_text_formats_options_and_prompt(self):
        prompt = utils.tvbench_doc_to_text(self.doc)
        self.assertIn("What is the person doing?", prompt)
        self.assertIn("A. Running", prompt)
        self.assertIn("B. Sitting", prompt)
        self.assertTrue(prompt.endswith("Answer with the option letter only."))

    def test_doc_to_target_maps_answer_to_option_letter(self):
        self.assertEqual(utils.tvbench_doc_to_target(self.doc), "B")

    def test_process_results_accepts_option_letter(self):
        result = utils.tvbench_process_results(self.doc, ["B"])
        self.assertEqual(result["tvbench_acc"], 1.0)

    def test_process_results_accepts_option_text(self):
        result = utils.tvbench_process_results(self.doc, ["The answer is Sitting."])
        self.assertEqual(result["tvbench_acc"], 1.0)

    def test_doc_to_visual_returns_resolved_or_fallback_path(self):
        with patch("lmms_eval.tasks.tvbench.utils.os.path.exists", return_value=False):
            visual_paths = utils.tvbench_doc_to_visual(self.doc)
        self.assertEqual(len(visual_paths), 1)
        self.assertTrue(visual_paths[0].endswith("sample_video.mp4"))


if __name__ == "__main__":
    unittest.main()

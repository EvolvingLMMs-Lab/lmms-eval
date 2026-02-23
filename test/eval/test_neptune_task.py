import unittest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.neptune import utils as neptune_utils


class TestNeptuneTaskRegistration(unittest.TestCase):
    def test_neptune_tasks_are_registered(self):
        task_manager = TaskManager()
        expected_tasks = {
            "neptune",
            "neptune_full_v",
            "neptune_mma_v",
            "neptune_mmh_v",
            "neptune_full_i",
            "neptune_mma_i",
            "neptune_mmh_i",
        }

        missing_tasks = expected_tasks.difference(task_manager.all_tasks)
        self.assertFalse(missing_tasks, f"Missing Neptune tasks: {sorted(missing_tasks)}")


class TestNeptuneMetrics(unittest.TestCase):
    def _build_doc(self, answer_id=2, question_type="Counting"):
        return {
            "key": "demo/1",
            "video_path": "demo.mp4",
            "question": "How many cats are visible?",
            "answer_choice_0": "One",
            "answer_choice_1": "Two",
            "answer_choice_2": "Three",
            "answer_choice_3": "Four",
            "answer_choice_4": "Five",
            "answer_id": answer_id,
            "question_type": question_type,
        }

    def test_process_results_parses_choice_letter(self):
        doc = self._build_doc(answer_id=2)
        result = neptune_utils.neptune_process_results(doc, ["The correct answer is C."])

        self.assertEqual(result["neptune_acc"]["parsed_pred"], "C")
        self.assertTrue(result["neptune_acc"]["is_correct"])

    def test_aggregate_results_returns_overall_accuracy(self):
        results = [
            {"question_type": "Counting", "is_correct": True},
            {"question_type": "Counting", "is_correct": False},
            {"question_type": "Summarization", "is_correct": True},
        ]

        score = neptune_utils.neptune_aggregate_results(results)
        self.assertAlmostEqual(score, 2 / 3)


if __name__ == "__main__":
    unittest.main()

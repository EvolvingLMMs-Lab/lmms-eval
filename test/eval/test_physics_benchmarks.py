import io
import os
import tempfile
import unittest
from unittest.mock import patch

import datasets
from PIL import Image

from lmms_eval.tasks.physbench import utils as physbench_utils
from lmms_eval.tasks.physgame import utils as physgame_utils
from lmms_eval.tasks.physics_rw import utils as physics_rw_utils
from lmms_eval.tasks.physreason import utils as physreason_utils
from tools.prepare_physics_benchmarks import (
    _normalize_explanation_steps,
    _normalize_image_captions,
    _normalize_step_analysis,
    _normalize_theorems,
)


class TestPhysicsBenchmarks(unittest.TestCase):
    def test_physbench_filters_unscored_rows_and_resolves_ordered_media(self):
        dataset = datasets.Dataset.from_list(
            [
                {"answer": "", "idx": 1},
                {"answer": "B", "idx": 2},
            ]
        )
        filtered = physbench_utils.physbench_process_docs(dataset)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["idx"], 2)

        with tempfile.TemporaryDirectory() as cache_dir:
            media_names = ["clip.mp4", "frame.png"]
            for media_name in media_names:
                open(os.path.join(cache_dir, media_name), "wb").close()

            with patch.object(physbench_utils, "_get_cache_dir", return_value=cache_dir):
                visuals = physbench_utils.physbench_doc_to_visual({"file_name": media_names})

        self.assertEqual([os.path.basename(path) for path in visuals], media_names)

    def test_physgame_uses_normalized_options_and_video_path(self):
        doc = {
            "question_id": "abc123",
            "question": "What happened?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "answer": "C",
            "class_anno": "Mechanics",
            "subclass_anno": "Gravity",
            "video_path": "PhysGame-Benchmark/abc123.mp4",
        }
        prompt = physgame_utils.physgame_doc_to_text(doc)
        self.assertIn("A. One", prompt)
        self.assertIn("D. Four", prompt)

        result = physgame_utils.physgame_process_results(doc, ["The correct answer is (C)."])
        self.assertEqual(result["physgame_accuracy"]["pred_answer"], "C")

        with tempfile.TemporaryDirectory() as cache_dir:
            video_path = os.path.join(cache_dir, doc["video_path"])
            os.makedirs(os.path.dirname(video_path))
            open(video_path, "wb").close()
            with patch.object(physgame_utils, "_get_cache_dir", return_value=cache_dir):
                self.assertEqual(physgame_utils.physgame_doc_to_visual(doc), [video_path])

    def test_physics_rw_uses_normalized_domain_and_media_path(self):
        doc = {
            "id": "mechanics-0",
            "domain": "Mechanics",
            "instruction": "Will it fall?",
            "label": "yes",
            "video_path": "media/Mechanics/classification/video/example.mp4",
        }
        result = physics_rw_utils.physics_rw_process_results(doc, ["Yes, it will."])
        self.assertEqual(result["physics_rw_accuracy"]["pred_answer"], "yes")

        with tempfile.TemporaryDirectory() as cache_dir:
            video_path = os.path.join(cache_dir, doc["video_path"])
            os.makedirs(os.path.dirname(video_path))
            open(video_path, "wb").close()
            with patch.object(physics_rw_utils, "_get_cache_dir", return_value=cache_dir):
                self.assertEqual(physics_rw_utils.physics_rw_doc_to_visual(doc), [video_path])

    def test_physreason_preserves_multiple_images_and_scores_answers(self):
        image = Image.new("RGB", (2, 2), color="red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        doc = {
            "problem_id": "problem-1",
            "context": "A physics problem.",
            "sub_questions": ["Find x.", "Find y."],
            "answers": ["2 m", "3 s"],
            "difficulty": "easy",
            "images": [image, {"bytes": buffer.getvalue(), "path": "diagram.png"}],
        }

        visuals = physreason_utils.physreason_doc_to_visual(doc)
        self.assertEqual(len(visuals), 2)
        self.assertTrue(all(visual.mode == "RGB" for visual in visuals))

        prompt = physreason_utils.physreason_doc_to_text(doc)
        self.assertIn("(1) Find x.", prompt)
        self.assertIn("(2) Find y.", prompt)

        result = physreason_utils.physreason_process_results(doc, ["Answer (1): 2 m\nAnswer (2): 3 s"])
        self.assertEqual(result["physreason_accuracy"]["accuracy"], 1.0)

    def test_physreason_normalizes_inconsistent_source_keys(self):
        problem = {
            "image_caption": ["First.", "Second."],
            "explanation_steps": {"sub_question_1": {"step_2": "Later", "step_1": "First"}},
            "steps _analysis": {
                "step_1": {
                    "physical_theorem": "Newton's second law",
                    "result_quantity": [{"name": "force", "symbol": "F", "value": "ma", "unit": "N"}],
                }
            },
        }

        explanations = _normalize_explanation_steps(problem)
        analysis = _normalize_step_analysis(problem)

        self.assertEqual(_normalize_image_captions(problem), "First. Second.")
        self.assertEqual([item["step"] for item in explanations], ["step_1", "step_2"])
        self.assertEqual(analysis[0]["result_quantities"][0]["equation"], "")
        self.assertEqual(_normalize_theorems(problem, analysis), ["Newton's second law"])


if __name__ == "__main__":
    unittest.main()

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import datasets
import torch
from PIL import Image

from lmms_eval.models.simple.qwen2_5_vl import _build_video_content, _interleave_visual_content, _limit_video_inputs, _strip_visual_placeholders
from lmms_eval.tasks.physbench import utils as physbench_utils
from lmms_eval.tasks.physgame import utils as physgame_utils
from lmms_eval.tasks.physics_rw import utils as physics_rw_utils
from lmms_eval.tasks.physreason import utils as physreason_utils
from tools.prepare_physics_benchmarks import (
    PHYSICS_RW_PAPER_COUNTS,
    _download_url,
    _normalize_explanation_steps,
    _normalize_image_captions,
    _normalize_step_analysis,
    _normalize_theorems,
    build_physics_rw,
)
from tools.score_physreason_psas_a import _step_weights, _summarize


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

        scored = physbench_utils.physbench_process_results({"idx": 2, "answer": "B"}, ["B. Option"])
        self.assertEqual(scored["physbench_accuracy"]["score"], 1.0)
        verbose = physbench_utils.physbench_process_results({"idx": 2, "answer": "B"}, ["The answer is B."])
        self.assertEqual(verbose["physbench_accuracy"]["pred_answer"], "")

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
        self.assertIn("A: One", prompt)
        self.assertIn("D: Four", prompt)

        result = physgame_utils.physgame_process_results(doc, ["C"])
        self.assertEqual(result["physgame_accuracy"]["pred_answer"], "C")
        verbose_result = physgame_utils.physgame_process_results(doc, ["The correct answer is (C)."])
        self.assertEqual(verbose_result["physgame_accuracy"]["pred_answer"], "")

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
        self.assertEqual(result["physics_rw_macro_f1"]["pred_answer"], "yes")
        strict_result = physics_rw_utils.physics_rw_process_results(doc, ["I think the answer is yes."])
        self.assertEqual(strict_result["physics_rw_accuracy"]["pred_answer"], "")

        with tempfile.TemporaryDirectory() as cache_dir:
            video_path = os.path.join(cache_dir, doc["video_path"])
            os.makedirs(os.path.dirname(video_path))
            open(video_path, "wb").close()
            with patch.object(physics_rw_utils, "_get_cache_dir", return_value=cache_dir):
                self.assertEqual(physics_rw_utils.physics_rw_doc_to_visual(doc), [video_path])

        metric_rows = [
            {"pred_answer": "yes", "answer": "yes"},
            {"pred_answer": "yes", "answer": "no"},
            {"pred_answer": "no", "answer": "no"},
            {"pred_answer": "no", "answer": "yes"},
        ]
        self.assertEqual(physics_rw_utils.physics_rw_aggregate_macro_f1(metric_rows), 50.0)

    def test_physics_rw_builder_uses_the_paper_split(self):
        annotations = {}
        for domain, count in PHYSICS_RW_PAPER_COUNTS.items():
            rows = [
                {
                    "idx": index,
                    "video_path": f"video/{domain.lower()}_{index}.mp4",
                    "instruction": "Please answer yes or no only.",
                    "answer": "yes" if index % 2 else "no",
                }
                for index in range(count)
            ]
            annotations[domain] = rows

        # Mirror the 39 post-paper Mechanics rows in the current ModelScope
        # source. They must not silently change the published benchmark size.
        annotations["Mechanics"].extend(
            {
                "idx": index,
                "video_path": f"video/mechanics_extra_{index}.mp4",
                "instruction": "Please answer yes or no only.",
                "answer": "是",
            }
            for index in range(716, 755)
        )

        def source_rows(url):
            domain = next(domain for domain in PHYSICS_RW_PAPER_COUNTS if f"Physics-RW%2F{domain}%2F" in url)
            return annotations[domain]

        with patch("tools.prepare_physics_benchmarks._read_json_url", side_effect=source_rows):
            dataset = build_physics_rw()["test"]

        self.assertEqual(len(dataset), 1135)
        self.assertEqual(sum(row["domain"] == "Mechanics" for row in dataset), 716)
        self.assertTrue(all(row["label"] in {"yes", "no"} for row in dataset))

    def test_physics_rw_media_download_rejects_truncated_files(self):
        response = io.BytesIO(b"ab")
        response.headers = {"Content-Length": "4"}

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "video.mp4"
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "Incomplete download"):
                    _download_url("https://example.com/video.mp4", destination, retries=1)
            self.assertFalse(os.path.exists(destination))

    def test_physics_rw_media_download_rejects_checksum_mismatch(self):
        response = io.BytesIO(b"ab")
        response.headers = {"Content-Length": "2", "X-Linked-Etag": "0" * 64}

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "video.mp4"
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "Checksum mismatch"):
                    _download_url("https://example.com/video.mp4", destination, retries=1)
            self.assertFalse(os.path.exists(destination))

    def test_physics_rw_media_download_recovers_from_head_timeout(self):
        response = io.BytesIO(b"new")
        response.headers = {"Content-Length": "3"}

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "video.mp4"
            destination.write_bytes(b"cached")
            with patch("urllib.request.urlopen", side_effect=[TimeoutError(), response]):
                _download_url("https://example.com/video.mp4", destination, retries=1)
            self.assertEqual(destination.read_bytes(), b"new")

    def test_qwen_interleaves_plain_and_numbered_visual_markers(self):
        image_1 = {"type": "image", "image": "first"}
        image_2 = {"type": "image", "image": "second"}
        video = {"type": "video", "video": "clip"}

        content = _interleave_visual_content(
            "Transform <image> into <image>, then inspect <video>.",
            [image_1, image_2, video],
        )
        self.assertEqual([part["type"] for part in content], ["text", "image", "text", "image", "text", "video", "text"])
        self.assertIs(content[1], image_1)
        self.assertIs(content[3], image_2)
        self.assertIs(content[5], video)

        numbered = _interleave_visual_content("Compare <image 2> with <image 1>.", [image_1, image_2])
        self.assertIs(numbered[1], image_2)
        self.assertIs(numbered[3], image_1)
        self.assertEqual(_strip_visual_placeholders("<video> Q <image>"), " Q ")

    def test_qwen_caps_every_video_in_a_batch(self):
        videos = [torch.arange(20).reshape(10, 2), torch.arange(24).reshape(12, 2)]
        limited = _limit_video_inputs(videos, 4)
        self.assertEqual([video.shape[0] for video in limited], [4, 4])
        self.assertEqual(limited[0][0].tolist(), [0, 1])
        self.assertEqual(limited[0][-1].tolist(), [18, 19])

        fixed_frames = _build_video_content("clip.mp4", min_pixels=100352, max_pixels=602112, fps=None, nframes=8)
        self.assertEqual(fixed_frames["nframes"], 8)
        self.assertNotIn("fps", fixed_frames)

        sampled_by_fps = _build_video_content("clip.mp4", min_pixels=100352, max_pixels=345600, fps=2.0, nframes=None)
        self.assertEqual(sampled_by_fps["fps"], 2.0)
        self.assertNotIn("nframes", sampled_by_fps)

        with self.assertRaises(ValueError):
            _build_video_content("clip.mp4", min_pixels=100352, max_pixels=602112, fps=2.0, nframes=8)

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
            "image_1": image,
            "image_2": {"bytes": buffer.getvalue(), "path": "diagram.png"},
            "image_3": None,
            "image_4": None,
            "image_5": None,
        }

        visuals = physreason_utils.physreason_doc_to_visual(doc)
        self.assertEqual(len(visuals), 2)
        self.assertTrue(all(visual.mode == "RGB" for visual in visuals))
        self.assertEqual(len(physreason_utils.physreason_doc_to_visual({"images": [image]})), 1)

        prompt = physreason_utils.physreason_doc_to_text(doc)
        self.assertIn("(1) Find x.", prompt)
        self.assertIn("(2) Find y.", prompt)
        self.assertIn("continuously across all sub-questions", prompt)
        self.assertIn("one formula and its solution process in each step", prompt)
        self.assertIn("LaTeX notation ($)", prompt)

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

    def test_physreason_psas_a_uses_step_weights(self):
        problem = {
            "sub_questions": ["First?", "Second?"],
            "explanation_steps": [
                {"sub_question": "sub_question_1"},
                {"sub_question": "sub_question_2"},
                {"sub_question": "sub_question_2"},
            ],
        }
        weights = _step_weights(problem)
        self.assertEqual(weights, [1, 2])

        problems = {0: {"problem_id": "p0", "difficulty": "easy", "weights": weights}}
        records = {
            (0, 0): {"correct": False, "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
            (0, 1): {"correct": True, "usage": {"prompt_tokens": 20, "completion_tokens": 2}},
        }
        summary = _summarize(records, problems, expected_records=2)
        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["psas_a"], 200 / 3)
        self.assertEqual(summary["usage"], {"prompt_tokens": 30, "completion_tokens": 3})


if __name__ == "__main__":
    unittest.main()

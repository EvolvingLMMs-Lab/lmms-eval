import json
import os
import unittest
from unittest import mock

from lmms_eval.tasks.dive_bench import utils


class HighMotionFrameAlignmentTests(unittest.TestCase):
    @staticmethod
    def _doc(labels):
        return {
            "video": "clip",
            "qid": "q1",
            "type": "trajectory",
            "answer": json.dumps(labels),
            "question": (f"Move an object.\n\nWe consider all {len(labels)} frames in temporal order. For every frame t = 1…{{F}}, locate the hand.\n\nAnswer format: return a comma-separated list of {len(labels)} region names."),
        }

    def test_uniform_target_matches_endpoint_inclusive_eight_frame_sampling(self):
        labels = ["top"] * 94
        for index in (0, 13, 26, 39, 53, 66, 79, 93):
            labels[index] = "bottom"
        with mock.patch.dict(os.environ, {"DENSEVIDEO_HIGHMOTION_NUM_FRAMES": "8"}):
            target = utils.highmotion_doc_to_answer(self._doc(labels))
        self.assertEqual(target.split(","), ["bottom"] * 8)

    def test_prompt_replaces_the_full_frame_protocol(self):
        with mock.patch.dict(os.environ, {"DENSEVIDEO_HIGHMOTION_NUM_FRAMES": "8"}):
            prompt = utils.highmotion_doc_to_text(self._doc(["middle"] * 94))
        self.assertIn("exactly eight frames uniformly sampled", prompt)
        self.assertIn("including its first and last frames", prompt)
        self.assertIn("exactly eight labels", prompt)
        self.assertIn("exactly seven commas", prompt)
        self.assertIn("stop after the final frame's label", prompt)
        self.assertIn("until all eight frame slots are filled", prompt)
        self.assertIn("do not use vertical bars", prompt)
        self.assertNotRegex(prompt, r"\d")
        self.assertNotIn("all 94 frames", prompt)
        self.assertNotIn("{F}", prompt)
        self.assertNotIn("single word", prompt)

    def test_metrics_use_the_same_sampled_reference(self):
        labels = ["top", "middle", "bottom", "left", "right", "topleft", "topright", "bottomright"]
        doc = self._doc(labels)
        with mock.patch.dict(os.environ, {"DENSEVIDEO_HIGHMOTION_NUM_FRAMES": "8"}):
            target = utils.highmotion_doc_to_answer(doc)
            result = utils.highmotion_process_results(doc, [target])
        self.assertEqual(result["grid_acc"], 1.0)
        self.assertEqual(result["grid_ade"], 0.0)
        self.assertEqual(result["grid_fde"], 0.0)
        self.assertEqual(result["grid_transition_acc"], 1.0)
        self.assertEqual(result["token_f1"], 1.0)

    def test_token_f1_ignores_equivalent_grid_formatting(self):
        labels = ["top", "middle", "bottom", "left", "right", "topleft", "topright", "bottomright"]
        doc = self._doc(labels)
        predictions = (
            "top, middle, bottom, left, right, topleft, topright, bottomright",
            "r1c2,r2c2,r3c2,r2c1,r2c3,r1c1,r1c3,r3c3",
        )
        with mock.patch.dict(os.environ, {"DENSEVIDEO_HIGHMOTION_NUM_FRAMES": "8"}):
            for prediction in predictions:
                with self.subTest(prediction=prediction):
                    result = utils.highmotion_process_results(doc, [prediction])
                    self.assertEqual(result["grid_acc"], 1.0)
                    self.assertEqual(result["token_f1"], 1.0)

    def test_short_sequences_do_not_duplicate_labels(self):
        with mock.patch.dict(os.environ, {"DENSEVIDEO_HIGHMOTION_NUM_FRAMES": "8"}):
            target = utils.highmotion_doc_to_answer(self._doc(["top", "middle", "bottom"]))
        self.assertEqual(target, "top,middle,bottom")

    def test_invalid_frame_budget_fails_fast(self):
        for value in ("0", "bad"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"DENSEVIDEO_HIGHMOTION_NUM_FRAMES": value}):
                    with self.assertRaisesRegex(ValueError, "DENSEVIDEO_HIGHMOTION_NUM_FRAMES"):
                        utils.highmotion_doc_to_answer(self._doc(["top"] * 8))


if __name__ == "__main__":
    unittest.main()

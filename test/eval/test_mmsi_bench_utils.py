import unittest

from lmms_eval.tasks.mmsi_bench import utils


class TestMMSIBenchUtils(unittest.TestCase):
    def test_doc_to_text_handles_none_kwargs(self):
        doc = {"question": "Which option is correct?"}

        rendered = utils.msr_doc_to_text(doc, None)

        self.assertEqual(rendered, "Which option is correct?")

    def test_extract_single_choice_is_case_insensitive(self):
        score = utils.extract_single_choice_with_word_boundary("the answer is b", "b")

        self.assertEqual(score, 1.0)

    def test_process_results_normalizes_legacy_category_names(self):
        doc = {
            "id": "sample-1",
            "answer": "A",
            "question_type": "Positional Relationship (Cam.-Obj.)",
        }

        processed = utils.msr_process_results(doc, ["A"])

        self.assertIn("Positional Relationship (Cam.–Obj.)", processed)
        self.assertNotIn("Positional Relationship (Cam.-Obj.)", processed)


if __name__ == "__main__":
    unittest.main()

import os
import re
import ast
import json
import logging
from lmms_eval.api.metrics import levenshtein_distance
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

lmms_logger = logging.getLogger("lmms-eval")


def multidocvqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def multidocvqa_doc_to_visual(doc):
    return [doc[f"image_{i}"].convert("RGB") for i in range(1, 21) if doc[f"image_{i}"] is not None]


def multidocvqa_process_results(doc, results):
    pred_answer = results[0]
    answer = ast.literal_eval(doc["answers"])

    return {"anls": {"questionId": int(doc["questionId"]), "answer": answer, "pred_answer": pred_answer}, "accuracy": {"questionId": int(doc["questionId"]), "answer": answer, "pred_answer": pred_answer}}


def multidocvqa_aggregate_results_anls(results):
    keys = {k for result in results for k in result}
    results = {key: [result.get(key, None) for result in results] for key in keys}
    evaluator = Evaluator(case_sensitive=False)
    metric = evaluator.get_metrics(results["answer"], results["pred_answer"])

    return sum(metric["anls"]) / len(metric["anls"])


def multidocvqa_aggregate_results_accuracy(results):
    keys = {k for result in results for k in result}
    results = {key: [result.get(key, None) for result in results] for key in keys}
    evaluator = Evaluator(case_sensitive=False)
    metric = evaluator.get_metrics(results["answer"], results["pred_answer"])

    return sum(metric["accuracy"]) / len(metric["accuracy"])


def multidocvqa_process_test_results_for_submission(doc, results):
    answer = results[0]
    return {"submission": {"questionId": int(doc["questionId"]), "answer": answer, "answer_page": None}}


def multidocvqa_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file("multidocvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    lmms_logger.info(f"Results saved to {path}.")


##################
# Helper functions
##################


class Evaluator:
    def __init__(self, case_sensitive=False):
        self.case_sensitive = case_sensitive
        self.get_edit_distance = levenshtein_distance
        self.anls_threshold = 0.5

    def get_metrics(self, gt_answers, preds):
        batch_accuracy = []
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])

            batch_accuracy.append(self._calculate_accuracy(gt, pred))
            batch_anls.append(self._calculate_anls(gt, pred))

        return {"accuracy": batch_accuracy, "anls": batch_anls}

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()

    def _calculate_accuracy(self, gt, pred):
        if pred == "none":
            return 0

        for gt_elm in gt:
            if gt_elm == pred:
                return 1

        return 0

    def _calculate_anls(self, gt, pred):
        if len(pred) == 0:
            return 0

        if pred == "none":
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls


if __name__ == "__main__":
    print("-----------------")
    multidocvqa_aggregate_results_anls([{"questionId": 1, "answer": ["answer"], "pred_answer": "pred_answer"}, {"questionId": 2, "answer": ["nswer"], "pred_answer": "nswer"}])

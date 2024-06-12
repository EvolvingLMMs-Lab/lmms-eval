import logging
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.synthdog.donut_evaluator import JSONParseEvaluator

logger = logging.getLogger("lmms-eval")

evaluator = JSONParseEvaluator()


def synthdog_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def synthdog_doc_to_target(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [json.loads(doc["ground_truth"])["gt_parse"]["text_sequence"]]


def synthdog_process_results(doc, results):
    pred = {"output": results[0].lower().strip()}
    gt_ans = json.loads(doc["ground_truth"])["gt_parse"]

    predictions = []
    ground_truths = []
    accs = []

    score = evaluator.cal_acc(pred, gt_ans)

    accs.append(score)

    predictions.append(pred)
    ground_truths.append(gt_ans)

    return {
        "tree_edit_distance": {"score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def synthdog_aggregate_ted(results, args):
    final_score = 0
    for result in results:
        final_score += result["score"]
    return final_score

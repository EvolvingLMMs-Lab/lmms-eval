from collections import defaultdict
import os
from anls import anls_score


from loguru import logger as eval_logger

dir_name = os.path.dirname(os.path.abspath(__file__))

# 19 classes
eval_type_dict = {
    "Sensation": ["count", "color", "scene", "poster", "attribute_recognition", "ocr", "position"],
    "Cognition": ["calculation", "code", "translation", "math", "cross_instance_reason", "attribute_reason"],
    "Knowledge": ["celebrity", "chemistry", "physics", "biology", "landmark", "artwork"],
}


def conbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def conbench_doc_to_text(doc):
    question = doc["question"].strip()
    return question


def parse_pred_ans_NY(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def parse_pred_ans_choice(pred_ans):
    return pred_ans.replace(" ", "")[0]


def conbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred = pred.replace("\n", "").lower()
    # parser
    if doc["question_field"] == "N/Y":
        pred_ans = parse_pred_ans_NY(pred)
    elif doc["question_field"] == "Choices":
        pred_ans = parse_pred_ans_choice(pred)
    else:
        pred_ans = pred

    gt_ans = doc["answer"].lower()

    # score
    score = 1 if (doc["question_field"] == "Q/A" and anls_score(prediction=pred_ans, gold_labels=[gt_ans], threshold=0.95) >= 0.4) or (gt_ans == pred_ans) else 0
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {"ConScore_D": {"image_id": doc["image_id"], "question_field": doc["question_field"], "score": score}}


def conbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    summary = defaultdict(dict)
    for result in results:
        image_id = result["image_id"]
        score = result["score"]
        if image_id not in summary.keys():
            summary[image_id] = 0
        summary[image_id] += score

    cnt_con = 0
    for image_id, score in summary.items():
        if score == 3:
            cnt_con += 1

    print("Consistency Cases are ", cnt_con)
    cnt_con = cnt_con / (len(results) / 3)
    eval_logger.info(f"ConScore_D: {cnt_con:.2f}")
    return cnt_con
